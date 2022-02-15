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
    def __init__(self, input_path, output_path, arch, transform, icon_pretrained, split, task, num_patches):
        pid_splits = json.load(open(os.path.join(input_path, 'pid_splits.json')))
        self.data = pid_splits['%s_%s' % (task, split)] # len: 51766
        self.problems = json.load(open(os.path.join(input_path, 'problems.json')))
        self.input_path = input_path
        self.output_path = output_path
        self.arch = arch
        self.icon_pretrained = icon_pretrained
        self.transform = transform
        self.task = task
        self.num_patches = num_patches

    def crop_and_padding(self, img, padding=3):
        # Crop the image
        bbox = img.getbbox() # [left, top, right, bottom]
        img = img.crop(bbox)

        # Add padding spaces to the 4 sides of an image
        desired_size = max(img.size) + padding * 2
        if img.size[0] < desired_size or img.size[1] < desired_size:
            delta_w = desired_size - img.size[0]
            delta_h = desired_size - img.size[1]
            padding = (padding, padding, delta_w-padding, delta_h-padding)
            img = ImageOps.expand(img, padding, (255, 255, 255))

        return img

    def extract_patches(self, img, splits):
        patches = []
        w, h = img.size  # width, height
        for n in splits:
            dw, dh = w // n, h // n
            for j in range(n):
                for i in range(n):
                    bbox = dw * i, dh * j, dw * (i + 1), dh * (j + 1)
                    patch = img.crop(bbox)
                    patches.append(patch)
        return patches

    def resize_patches(self, patches):
        resized_patches = []
        for patch in patches:
            patch = self.transform(patch)
            resized_patches.append(patch) # [3,224,224] * num_patches
        patch_input = torch.stack(resized_patches, dim=0) # [num_patches,3,224,224]
        return patch_input

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pid = self.data[index]

        local_split = self.problems[pid]['split']
        assert local_split in ['train', 'val', 'test']

        # process images
        img_file = os.path.join(self.input_path, 'iconqa', local_split, self.task, pid, "image.png")
        img = Image.open(img_file)
        img = img.convert('RGB')
        img = self.crop_and_padding(img)

        # obtain patches from the image
        if self.num_patches == 25:
            patches = self.extract_patches(img, [5])
        elif self.num_patches == 36:
            patches = self.extract_patches(img, [6])
        elif self.num_patches == 14:
            patches = self.extract_patches(img, [1, 2, 3])
        elif self.num_patches == 30:
            patches = self.extract_patches(img, [1, 2, 3, 4])
        elif self.num_patches == 79:
            patches = self.extract_patches(img, [1, 2, 3, 4, 7])
        
        # num_patches * [3,224,224] -> [num_patches,3,224,224]
        patch_input = self.resize_patches(patches)

        # convert to Tensor so we can batch it
        img_id = torch.LongTensor([int(pid)])

        return patch_input, img_id


def preprocess_images(input_path, output_path, arch, layer, icon_pretrained, split, task, patch_split):
    """
    Generate image patch embeddings for IconQA images.
    """
    num_patches = patch_split

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
                                                icon_pretrained= icon_pretrained,
                                                num_patches=patch_split,
                                                split=split, task=task),
                                  batch_size=8, shuffle=False, num_workers=4)

    # model
    model = image_models.get_image_model(arch, layer, icon_pretrained)
    model = model.eval().to(device)
    print("ConvNet Model:", arch, layer)

    # generate image embeddings
    embeddings = {}

    print("Starting:")
    with torch.no_grad():

        print("total image batches:", len(data_loader))
        for img_patches, img_id in tqdm(data_loader, total=len(data_loader)):
            img_patches = img_patches.to(device) # [batch,num_patches,3,224,224]

            model_input = img_patches.view(-1,3,224,224) # [batch*num_patches,3,224,224]
            embedding = model(model_input) # [batch*num_patches,2048,1,1]
            embedding = embedding.squeeze(3).squeeze(2) # [batch*num_patches,2048]
            embedding = embedding.view(-1,num_patches,2048) # [batch,num_patches,2048]
            assert list(embedding.size())[1:] == [num_patches,2048]
            #print("embedding size", embedding.size()) # pool5: [batch,num_patches,2048]

            for idx in range(img_patches.size(0)):
                assert list(embedding[idx, ...].size()) == [num_patches,2048]
                embeddings[img_id[idx].item()] = embedding[idx, ...].cpu()

    print("Computing image embeddings, Done!")

    # save results
    output_path = os.path.join(output_path, "{}_{}_{}".format(arch, layer, patch_split))
    if icon_pretrained:
        output_path = output_path + "_icon"
    print("final output path:", output_path)
    os.makedirs(output_path, exist_ok=True)

    print("Saving image embedddings:")
    if not icon_pretrained:
        image_embedding_file = os.path.join(output_path,
                                        "iconqa_{0}_{1}_{2}_{3}_{4}.pth".format(split, task, arch, layer, patch_split))
    elif icon_pretrained:
        image_embedding_file = os.path.join(output_path,
                                        "iconqa_{0}_{1}_{2}_{3}_{4}_icon.pth".format(split, task, arch, layer, patch_split))
    print("Saved to {}".format(image_embedding_file))
    torch.save(embeddings, image_embedding_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Standalone utility to preprocess ICONQA images")
    # input and output
    parser.add_argument("--input_path", default="../data/iconqa_data",
                        help="path to the root directory of images")
    parser.add_argument("--output_path", default="../data/iconqa_data",
                        help="path to image features")
    # image model
    parser.add_argument("--arch", default="resnet101")
    parser.add_argument("--layer", default="pool5")
    parser.add_argument("--icon_pretrained", default=False, help='use the icon pretrained model or not')
    parser.add_argument("--patch_split", type=int, default=30, choices=[14,25,30,36,79])
    # tasks and splits
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test", "trainval", "minitrain", "minival", "minitest"])
    parser.add_argument("--task", default="fill_in_blank",
                        choices=["fill_in_blank", "choose_txt", "choose_img"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # GPU
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    # manual settings
    tasks = ["fill_in_blank", "choose_txt", "choose_img"]
    splits = ["test", "val", "train"]
    # splits = ["minival", "minitrain", "test", "val", "train"] # "minival", "minitrain" for quick checking

    for task in tasks:
        for split in splits:
            args.task, args.split = task, split
            print("\n----------------- Processing {} for {} -----------------".format(args.task, args.split))

            # preprocess images
            for arg in vars(args):
                print('%s: %s' % (arg, getattr(args, arg)))
            preprocess_images(args.input_path, args.output_path, args.arch, args.layer, 
                              args.icon_pretrained, args.split, args.task, args.patch_split)
