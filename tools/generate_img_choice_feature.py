import argparse
import json
import os
import warnings
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
from torch.utils import data
from torchvision import transforms
import image_models

warnings.filterwarnings("ignore")


class ICONQADataset(data.Dataset):
    def __init__(self, input_path, output_path, arch, transform, icon_pretrained, split, task):
        pid_splits = json.load(open(os.path.join(input_path, 'pid_splits.json')))
        self.data = pid_splits['%s_%s' % (task, split)] # len: 51766
        self.problems = json.load(open(os.path.join(input_path, 'problems.json')))
        self.input_path = input_path
        self.output_path = output_path
        self.arch = arch
        self.icon_pretrained = icon_pretrained
        self.transform = transform
        self.task = task

    def crop_margin(self, img_fileobj):
        ivt_image = ImageOps.invert(img_fileobj)
        bbox = ivt_image.getbbox()  # [left, top, right, bottom]
        cropped_image = img_fileobj.crop(bbox)

        return cropped_image

    def add_padding(self, img, padding=2):
        """Add borders to the 4 sides of an image"""
        desired_size = max(img.size) + padding * 2

        if img.size[0] < desired_size or img.size[1] < desired_size:
            delta_w = desired_size - img.size[0]
            delta_h = desired_size - img.size[1]
            padding = (max(delta_w // 2, 0), max(delta_h // 2, 0),
                       max(delta_w - (delta_w // 2), 0), max(delta_h - (delta_h // 2), 0))
            img = ImageOps.expand(img, padding, (255, 255, 255))
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pid = self.data[index]
        img_id = torch.LongTensor([int(pid)]) # convert to Tensor so we can batch it

        choices = self.problems[pid]['choices']
        local_split = self.problems[pid]['split']
        assert local_split in ['train', 'val', 'test']

        img_choices = []
        for choice in choices:
            img_file = os.path.join(self.input_path, 'iconqa', local_split, self.task, pid, choice)
            img = Image.open(img_file)
            img = img.convert('RGB')

            # We need crop and pad the choice images for icon_pretrained mode
            # Because the icon_pretrained model is pretrained on our icon data
            if self.icon_pretrained:
                img = self.crop_margin(img)
                img = self.add_padding(img)

            if self.transform is not None:
                img = self.transform(img) # [3,224,224]
            img_choices.append(img)

        assert 2 <= len(img_choices) <= 5
        return img_choices, img_id


def preprocess_images(input_path, output_path, arch, layer, icon_pretrained, split, task):
    """
    Generate image choice embeddings for IconQA images.
    """
    # image transformer
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    data_loader = data.DataLoader(ICONQADataset(input_path, output_path,
                                                arch=arch, transform=transform,
                                                icon_pretrained=icon_pretrained,
                                                split=split, task=task),
                                  batch_size=1, shuffle=False, num_workers=1)

    # model
    model = image_models.get_image_model(arch, layer, icon_pretrained)
    model = model.eval().to(device)
    print("ConvNet Model:", arch, layer)

    # generate image embeddings
    embeddings = {}
    print("Starting:")

    with torch.no_grad():
        print("total images:", len(data_loader))
        for img_choices, img_id in tqdm(data_loader, total=len(data_loader)):
            choice_max_num = 5
            v_dim = 2048
            choice_embedding = torch.FloatTensor(choice_max_num, v_dim).zero_()

            for i, img in enumerate(img_choices):
                img = img.to(device)
                embedding = model(img) # [1,2048,1,1]
                embedding = embedding.squeeze(3).squeeze(2).squeeze(0) # [2048]
                assert list(embedding.size()) == [2048] # for choice images, only pool5 with 2048-D is allowed
                choice_embedding[i, :].copy_(embedding)
            embeddings[img_id.item()] = choice_embedding.cpu()

    print("Computing image embeddings, Done!")

    # save results
    output_path = os.path.join(output_path, "{}_{}".format(arch, layer))
    if icon_pretrained:
        output_path = output_path + "_icon"
    print("final output path:", output_path)
    os.makedirs(output_path, exist_ok=True)

    print("Saving image embedddings:")
    if not icon_pretrained:
        image_embedding_file = os.path.join(output_path,
                                        "iconqa_{0}_{1}_{2}_{3}.pth".format(split, task, arch, layer))
    elif icon_pretrained:
        image_embedding_file = os.path.join(output_path,
                                        "iconqa_{0}_{1}_{2}_{3}_icon.pth".format(split, task, arch, layer))
    print("Saved to {}".format(image_embedding_file))
    torch.save(embeddings, image_embedding_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Standalone utility to preprocess ICONQA choice images")
    # input and output
    parser.add_argument("--input_path", default="../data/iconqa_data",
                        help="path to the root directory of images")
    parser.add_argument("--output_path", default="../data/iconqa_data",
                        help="path to image features")
    # image model
    parser.add_argument("--arch", default="resnet101", choices=["resnet101"])
    parser.add_argument("--layer", default="pool5", choices=["pool5"], help="pool5 with 2048-D")
    parser.add_argument("--icon_pretrained", default=False, help='use the icon pretrained model or not')
    # tasks and splits
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test", "trainval", "minitrain", "minival", "minitest"])
    parser.add_argument("--task", default="choose_img", choices=["choose_img"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # GPU
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    # manual settings
    splits = ["test", "val", "train"]
    # splits = ["minival", "minitrain", "test", "val", "train"] # "minival", "minitrain" for quick checking

    for split in splits:
        args.split = split
        print("\n----------------- Processing {} for {} -----------------\n".format(args.task, args.split))

        # preprocess images
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)))
        preprocess_images(args.input_path, args.output_path, args.arch, args.layer, 
                          args.icon_pretrained, args.split, args.task)
