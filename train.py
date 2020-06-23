from models import *
from data import *
from utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import time
import datetime
import os
import PIL.Image as Image
from apex import amp
from torch.utils.data import DataLoader

device = (0 if torch.cuda.is_available() else 'cpu')

seed = 1
torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
np.random.seed(seed)


# wrapper class for training/pretraining of models
class Model:
    def __init__(self, model, dataset, model_name, pretrained=False, pretrained_SISRNet=None, pretrained_RegNet=None):
        if pretrained:
            assert pretrained_SISRNet or pretrained_RegNet, 'neither block is pretrained ??'
        self.num_epochs = 500
        self.batch_size = 32
        self.learning_rate = 0.001

        self.model = model
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.model_name = model_name
        self.pretrained = pretrained
        self.pretrained_SISRNet, self.pretrained_RegNet = pretrained_SISRNet, pretrained_RegNet
        # if we want to use AMP (automatic mixed precision) or train on multiple GPUs
        # self.model, self.opt = amp.initialize(self.model, self.opt, opt_level="O1")
        # self.model = nn.DataParallel(self.model)

    def train(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        # if we want to use AMP (automatic mixed precision) or train on multiple GPUs
        # self.model, opt = amp.initialize(self.model, opt, opt_level="O1")
        # self.model = nn.DataParallel(self.model)

        self.model.train()
        l1_criterion = nn.L1Loss() # we need to implement the actual loss used in DeepSUM

        # writer for each tensorboard session, each differentiated by the current time/date for keeping track
        writer = SummaryWriter(log_dir='runs/' + self.model_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        num_iters = 1

        data_iter = iter(self.dataloader)
        batch = data_iter.next()
        # set a single batch aside so we can plot how the model transforms it over time on tensorboard
        lr_batch_STATIC, hr_batch_STATIC = batch['LR'].to(device).float(), batch['HR'].to(device).float()


        for epoch in range(self.num_epochs):
            print(' ----- Epoch {} ----- '.format(epoch))

            data_iter = iter(self.dataloader)
            i = 0

            while i < len(self.dataloader):
                batch = data_iter.next()
                lr_batch, hr_batch = batch['LR'].to(device).float(), batch['HR'].to(device).float()

                self.model.train()
                self.model.zero_grad()

                # training stuff
                loss = ...
                loss.backward()
                opt.step()

                writer.add_scalar('Loss/Loss', loss, num_iters)


                i += 1
                num_iters += 1
                if num_iters % 50 == 0 or epoch == self.num_epochs - 1:
                    fig = plot_images(self.model, lr_batch_STATIC, hr_batch_STATIC, self.batch_size)
                    writer.add_figure('Example Images', fig, global_step=num_iters)

            # checkpoint
            if epoch % 1 == 0 or epoch == self.num_epochs - 1:
                root = 'models/' + model_name + '/'
                if not os.path.exists(root):
                    os.mkdir(root)
                torch.save(gen.state_dict(), root + 'epoch' + str(epoch) + '.pt')

        return

    def pretrain_SISRNet(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        # if we want to use AMP (automatic mixed precision) or train on multiple GPUs
        # self.model, opt = amp.initialize(self.model, opt, opt_level="O1")
        # self.model = nn.DataParallel(self.model)

        self.model.train()
        mse_criterion = nn.MSELoss() # we need to implement the actual loss used in DeepSUM

        # writer for each tensorboard session, each differentiated by the current time/date for keeping track
        writer = SummaryWriter(log_dir='runs/' + self.model_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        num_iters = 1

        data_iter = iter(self.dataloader)
        batch = data_iter.next()
        # set a single batch aside so we can plot how the model transforms it over time on tensorboard
        lr_batch_STATIC, hr_batch_STATIC = batch['LR'].to(device).float(), batch['HR'].to(device).float()


        for epoch in tqdm(range(self.num_epochs)):
            print(' ----- Epoch {} ----- '.format(epoch))

            data_iter = iter(self.dataloader)
            i = 0

            while i < len(self.dataloader):
                batch = data_iter.next()
                lr_batch, hr_batch = batch['LR'].to(device).float(), batch['HR'].to(device).float()
                bs, n_channels, depth, W, H = lr_batch.size()
                lr_batch_singles = lr_batch[:, :, 0:1, :, :] # take first image only so we can pretrain with pure SISR

                self.model.train()
                self.model.zero_grad()

                # training stuff
                projection = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0).to(device)

                features = self.model(lr_batch_singles) # [b, 64, 1, W, H]
                out_images = projection(features.view(-1, 64, W, H)) # [b, 1, W, H]

                loss = mse_criterion(out_images, hr_batch)
                loss.backward()
                opt.step()

                writer.add_scalar('Loss/Loss', loss, num_iters)


                i += 1
                num_iters += 1
                if num_iters % 50 == 0 or epoch == self.num_epochs - 1:
                    fig = plot_images(self.model, lr_batch_STATIC, hr_batch_STATIC, self.batch_size)
                    writer.add_figure('Example Images', fig, global_step=num_iters)

            # checkpoint
            if epoch % 1 == 0 or epoch == self.num_epochs - 1:
                root = 'models/' + model_name + '/'
                if not os.path.exists(root):
                    os.mkdir(root)
                torch.save(gen.state_dict(), root + 'epoch' + str(epoch) + '.pt')

        return

    def pretrain_RegNet(self):
        return



def run():
    torch.multiprocessing.freeze_support() # if using multiprocessing (num_workers > 0), i think this is necessary? lmao
    model = SISRNet().to(device)

    data = TrainNIRDataset()
    model_wrapper = Model(model, data, 'attempt1', pretrained=False)
    model_wrapper.pretrain_SISRNet()

if __name__ == '__main__':
    run()
