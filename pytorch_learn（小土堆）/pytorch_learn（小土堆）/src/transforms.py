from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


img_path = "../dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer = SummaryWriter("../logs")

writer.add_image("tensor", tensor_img)

writer.close()
