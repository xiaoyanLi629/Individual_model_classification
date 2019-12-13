import torch
import torch.nn as nn


class CNN0(nn.Module):
	def __init__(self):
		super(CNN0, self).__init__()

		self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
			nn.Conv2d(
				in_channels=1,  # input height
				out_channels=24,  # n_filters
				kernel_size=5,  # filter size
				stride=1,  # filter movement/step
				padding=2,
				# if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
			),  # output shape (16, 28, 28)
			nn.ReLU(),  # activation
			nn.MaxPool2d(kernel_size=5),  # choose max value in 2x2 area, output shape (16, 14, 14)
		)
		self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
			nn.Conv2d(24, 48, 5, 1, 2),  # output shape (32, 14, 14)
			nn.ReLU(),  # activation
			nn.MaxPool2d(5),  # output shape (32, 7, 7)
		)
		self.linear1 = nn.Linear(48, 512)
		self.linear2 = nn.Linear(512, 32)
		self.out = nn.Linear(32, 10)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		x = self.relu(x)
		x = self.linear1(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		x = self.linear2(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		out = self.out(x)
		return out, x  # return x for visualization


class CNN1(nn.Module):
	def __init__(self):
		super(CNN1, self).__init__()

		self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
			nn.Conv2d(
				in_channels=1,  # input height
				out_channels=24,  # n_filters
				kernel_size=5,  # filter size
				stride=1,  # filter movement/step
				padding=2,
				# if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
			),  # output shape (16, 28, 28)
			nn.ReLU(),  # activation
			nn.MaxPool2d(kernel_size=5),  # choose max value in 2x2 area, output shape (16, 14, 14)
		)
		self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
			nn.Conv2d(24, 48, 5, 1, 2),  # output shape (32, 14, 14)
			nn.ReLU(),  # activation
			nn.MaxPool2d(5),  # output shape (32, 7, 7)
		)
		self.linear1 = nn.Linear(48, 512)
		self.linear2 = nn.Linear(512, 32)
		self.out = nn.Linear(32, 10)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		x = self.relu(x)
		x = self.linear1(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		x = self.linear2(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		out = self.out(x)
		return out, x  # return x for visualization


class CNN2(nn.Module):
	def __init__(self):
		super(CNN2, self).__init__()

		self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
			nn.Conv2d(
				in_channels=1,  # input height
				out_channels=24,  # n_filters
				kernel_size=5,  # filter size
				stride=1,  # filter movement/step
				padding=2,
				# if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
			),  # output shape (16, 28, 28)
			nn.ReLU(),  # activation
			nn.MaxPool2d(kernel_size=5),  # choose max value in 2x2 area, output shape (16, 14, 14)
		)
		self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
			nn.Conv2d(24, 48, 5, 1, 2),  # output shape (32, 14, 14)
			nn.ReLU(),  # activation
			nn.MaxPool2d(5),  # output shape (32, 7, 7)
		)
		self.linear1 = nn.Linear(48, 512)
		self.linear2 = nn.Linear(512, 32)
		self.out = nn.Linear(32, 10)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		x = self.relu(x)
		x = self.linear1(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		x = self.linear2(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		out = self.out(x)
		return out, x  # return x for visualization


class CNN3(nn.Module):
	def __init__(self):
		super(CNN3, self).__init__()

		self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
			nn.Conv2d(
				in_channels=1,  # input height
				out_channels=24,  # n_filters
				kernel_size=5,  # filter size
				stride=1,  # filter movement/step
				padding=2,
				# if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
			),  # output shape (16, 28, 28)
			nn.ReLU(),  # activation
			nn.MaxPool2d(kernel_size=5),  # choose max value in 2x2 area, output shape (16, 14, 14)
		)
		self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
			nn.Conv2d(24, 48, 5, 1, 2),  # output shape (32, 14, 14)
			nn.ReLU(),  # activation
			nn.MaxPool2d(5),  # output shape (32, 7, 7)
		)
		self.linear1 = nn.Linear(48, 512)
		self.linear2 = nn.Linear(512, 32)
		self.out = nn.Linear(32, 10)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		x = self.relu(x)
		x = self.linear1(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		x = self.linear2(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		out = self.out(x)
		return out, x  # return x for visualization


class CNN4(nn.Module):
	def __init__(self):
		super(CNN4, self).__init__()

		self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
			nn.Conv2d(
				in_channels=1,  # input height
				out_channels=24,  # n_filters
				kernel_size=5,  # filter size
				stride=1,  # filter movement/step
				padding=2,
				# if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
			),  # output shape (16, 28, 28)
			nn.ReLU(),  # activation
			nn.MaxPool2d(kernel_size=5),  # choose max value in 2x2 area, output shape (16, 14, 14)
		)
		self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
			nn.Conv2d(24, 48, 5, 1, 2),  # output shape (32, 14, 14)
			nn.ReLU(),  # activation
			nn.MaxPool2d(5),  # output shape (32, 7, 7)
		)
		self.linear1 = nn.Linear(48, 512)
		self.linear2 = nn.Linear(512, 32)
		self.out = nn.Linear(32, 10)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		x = self.relu(x)
		x = self.linear1(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		x = self.linear2(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		out = self.out(x)
		return out, x  # return x for visualization


class CNN5(nn.Module):
	def __init__(self):
		super(CNN5, self).__init__()

		self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
			nn.Conv2d(
				in_channels=1,  # input height
				out_channels=24,  # n_filters
				kernel_size=5,  # filter size
				stride=1,  # filter movement/step
				padding=2,
				# if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
			),  # output shape (16, 28, 28)
			nn.ReLU(),  # activation
			nn.MaxPool2d(kernel_size=5),  # choose max value in 2x2 area, output shape (16, 14, 14)
		)
		self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
			nn.Conv2d(24, 48, 5, 1, 2),  # output shape (32, 14, 14)
			nn.ReLU(),  # activation
			nn.MaxPool2d(5),  # output shape (32, 7, 7)
		)
		self.linear1 = nn.Linear(48, 512)
		self.linear2 = nn.Linear(512, 32)
		self.out = nn.Linear(32, 10)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		x = self.relu(x)
		x = self.linear1(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		x = self.linear2(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		out = self.out(x)
		return out, x  # return x for visualization


class CNN6(nn.Module):
	def __init__(self):
		super(CNN6, self).__init__()

		self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
			nn.Conv2d(
				in_channels=1,  # input height
				out_channels=24,  # n_filters
				kernel_size=5,  # filter size
				stride=1,  # filter movement/step
				padding=2,
				# if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
			),  # output shape (16, 28, 28)
			nn.ReLU(),  # activation
			nn.MaxPool2d(kernel_size=5),  # choose max value in 2x2 area, output shape (16, 14, 14)
		)
		self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
			nn.Conv2d(24, 48, 5, 1, 2),  # output shape (32, 14, 14)
			nn.ReLU(),  # activation
			nn.MaxPool2d(5),  # output shape (32, 7, 7)
		)
		self.linear1 = nn.Linear(48, 512)
		self.linear2 = nn.Linear(512, 32)
		self.out = nn.Linear(32, 10)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		x = self.relu(x)
		x = self.linear1(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		x = self.linear2(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		out = self.out(x)
		return out, x  # return x for visualization


class CNN7(nn.Module):
	def __init__(self):
		super(CNN7, self).__init__()

		self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
			nn.Conv2d(
				in_channels=1,  # input height
				out_channels=24,  # n_filters
				kernel_size=5,  # filter size
				stride=1,  # filter movement/step
				padding=2,
				# if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
			),  # output shape (16, 28, 28)
			nn.ReLU(),  # activation
			nn.MaxPool2d(kernel_size=5),  # choose max value in 2x2 area, output shape (16, 14, 14)
		)
		self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
			nn.Conv2d(24, 48, 5, 1, 2),  # output shape (32, 14, 14)
			nn.ReLU(),  # activation
			nn.MaxPool2d(5),  # output shape (32, 7, 7)
		)
		self.linear1 = nn.Linear(48, 512)
		self.linear2 = nn.Linear(512, 32)
		self.out = nn.Linear(32, 10)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		x = self.relu(x)
		x = self.linear1(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		x = self.linear2(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		out = self.out(x)
		return out, x  # return x for visualization


class CNN8(nn.Module):
	def __init__(self):
		super(CNN8, self).__init__()

		self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
			nn.Conv2d(
				in_channels=1,  # input height
				out_channels=24,  # n_filters
				kernel_size=5,  # filter size
				stride=1,  # filter movement/step
				padding=2,
				# if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
			),  # output shape (16, 28, 28)
			nn.ReLU(),  # activation
			nn.MaxPool2d(kernel_size=5),  # choose max value in 2x2 area, output shape (16, 14, 14)
		)
		self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
			nn.Conv2d(24, 48, 5, 1, 2),  # output shape (32, 14, 14)
			nn.ReLU(),  # activation
			nn.MaxPool2d(5),  # output shape (32, 7, 7)
		)
		self.linear1 = nn.Linear(48, 512)
		self.linear2 = nn.Linear(512, 32)
		self.out = nn.Linear(32, 10)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		x = self.relu(x)
		x = self.linear1(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		x = self.linear2(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		out = self.out(x)
		return out, x  # return x for visualization


class CNN9(nn.Module):
	def __init__(self):
		super(CNN9, self).__init__()

		self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
			nn.Conv2d(
				in_channels=1,  # input height
				out_channels=24,  # n_filters
				kernel_size=5,  # filter size
				stride=1,  # filter movement/step
				padding=2,
				# if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
			),  # output shape (16, 28, 28)
			nn.ReLU(),  # activation
			nn.MaxPool2d(kernel_size=5),  # choose max value in 2x2 area, output shape (16, 14, 14)
		)
		self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
			nn.Conv2d(24, 48, 5, 1, 2),  # output shape (32, 14, 14)
			nn.ReLU(),  # activation
			nn.MaxPool2d(5),  # output shape (32, 7, 7)
		)
		self.linear1 = nn.Linear(48, 512)
		self.linear2 = nn.Linear(512, 32)
		self.out = nn.Linear(32, 10)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		x = self.relu(x)
		x = self.linear1(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		x = self.linear2(x)
		torch.nn.Dropout(0.2)
		x = self.relu(x)
		out = self.out(x)
		return out, x  # return x for visualization
