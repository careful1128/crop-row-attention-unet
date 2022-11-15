import joblib,copy
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch,sys
from tqdm import tqdm

from collections import OrderedDict
from lib.visualize import save_img,group_images,concat_result
import os
import argparse
from lib.logger import Logger, Print_Logger
from lib.extract_patches import *
from os.path import join
from lib.dataset import TestDataset
from lib.metrics import Evaluate
# import models
from lib.common import setpu_seed,dict_round
# from config import parse_args
from lib.pre_processing import my_PreProc

from unet import UNet
import torch.nn.functional as F
import models


setpu_seed(2021)

class Test():
    def __init__(self, args):
        self.args = args
        assert (args.stride_height <= args.test_patch_height and args.stride_width <= args.test_patch_width)
        # save path
        self.path_experiment = join(args.input, "preds/")

        self.patches_imgs_test, self.test_imgs, self.test_masks, self.test_FOVs, self.new_height, self.new_width = get_data_test_overlap(
            test_data_path_list=args.input,
            patch_height=args.test_patch_height,
            patch_width=args.test_patch_width,
            stride_height=args.stride_height,
            stride_width=args.stride_width
        )
        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=3)

    # Inference prediction process
    def inference(self, net):
        net.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = inputs.cuda()
                outputs = net(inputs)
                outputs = F.softmax(outputs, dim=1)[:, 1].cpu().squeeze().detach().numpy()
                # outputs = outputs[:,1].data.cpu().numpy()
                preds.append(outputs)
        predictions = np.concatenate(preds, axis=0)
        self.pred_patches = np.expand_dims(predictions,axis=1)
        
    # Evaluate ate and visualize the predicted images
    def evaluate(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## restore to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        log = eval.save_all_result(plot_curve=True,save_name="performance.txt")
        # save labels and probs for plot ROC and PR curve when k-fold Cross-validation
        np.save('{}/result.npy'.format(self.path_experiment), np.asarray([y_true, y_scores]))
        return dict_round(log, 6)

    # save segmentation imgs
    def save_segmentation_result(self):
        # img_path_list, _, _ = load_file_path_txt(self.args.test_data_path_list)
        img_path_list, _, _ = load_file_path_txt(self.args.input)
        img_name_list = [item.split('/')[-1].split('.')[0] for item in img_path_list]

        kill_border(self.pred_imgs, self.test_FOVs) # only for visualization
        self.save_img_path = join(self.path_experiment,'result_img')
        if not os.path.exists(join(self.save_img_path)):
            os.makedirs(self.save_img_path)
        self.test_imgs = my_PreProc(self.test_imgs) # Uncomment to save the pre processed image
        for i in range(self.test_imgs.shape[0]):
            total_img = concat_result(self.test_imgs[i],self.pred_imgs[i],self.test_masks[i])
            save_img(total_img,join(self.save_img_path, "Result_"+img_name_list[i]+'.png'))

    # Val on the test set at each epoch
    def val(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## recover to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        confusion,accuracy,specificity,sensitivity,precision = eval.confusion_matrix()
        log = OrderedDict([('val_auc_roc', eval.auc_roc()),
                           ('val_f1', eval.f1_score()),
                           ('val_acc', accuracy),
                           ('SE', sensitivity),
                           ('SP', specificity)])
        return dict_round(log, 6)

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./checkpoints/checkpoint_epoch5_mixed_our.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    # parser.add_argument('--test_patch_height', default=96)
    # parser.add_argument('--test_patch_width', default=96)
    # parser.add_argument('--stride_height', default=32)
    # parser.add_argument('--stride_width', default=32)
    parser.add_argument('--test_patch_height', default=192)
    parser.add_argument('--test_patch_width', default=192)
    parser.add_argument('--stride_height', default=64)
    parser.add_argument('--stride_width', default=64)
    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    args.input = args.input[0]
    save_path = join(args.input, "preds/")
    sys.stdout = Print_Logger(os.path.join(save_path, 'test_log.txt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # net = UNet(1, 2, args.bilinear)
    # net = models.UNetFamily.AttU_Net(1, 2).to(device)
    net = models.UNetFamily.AttEffU_Net(3, 2).to(device)
    cudnn.benchmark = True

    # Load checkpoint
    print('==> Loading checkpoint...')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    eval = Test(args)
    eval.inference(net)
    print(eval.evaluate())
    # eval.save_segmentation_result()
