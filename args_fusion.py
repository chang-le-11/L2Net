
class args():

	dataset = "E:/data/COCO/train2014/train2014/"
	epochs = 10
	batch_size = 16
	Datasize = 16321
	Step = Datasize // batch_size

	save_model_dir = "./model/model_con50"
	save_loss_dir = "models_gray/loss"


	HEIGHT = 256
	WIDTH = 256
	cuda = 1
	lr = 1e-4
	log_interval = 10
	resume = None
	trans_model_path = None
	is_para = False

	ssim_weight = [1, 10, 100, 1000, 10000]
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']



