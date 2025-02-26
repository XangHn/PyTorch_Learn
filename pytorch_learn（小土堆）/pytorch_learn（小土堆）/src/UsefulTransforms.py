from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("../logs")
img_path = "../dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
print(img)

trans_totensor = transforms.ToTensor()
trans_random = transforms.RandomCrop(512)
# trans_random = transforms.RandomCrop((500, 1000))
trans_compose = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()