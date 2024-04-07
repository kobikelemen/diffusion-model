import torch
import torch.nn as nn
import tqdm


class Trainer():

    def __init__(self, batch_size, timesteps, num_epochs, model, optimizer, loader_test, loader_train, img_c, img_w, img_h, rank, at_hat_list):
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.num_epochs = num_epochs
        self.model = model
        self.optimizer = optimizer
        self.loader_test = loader_test
        self.loader_train = loader_train
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.rank = rank
        self.at_hat_list = at_hat_list

    
    def _calc_xt(self, epsilon, t, x0):
        at_hat = self.at_hat_list[t]
        return torch.sqrt(at_hat.view(self.batch_size,1,1,1).repeat(1,self.img_c,1,1)) * x0 + torch.sqrt(1 - at_hat.view(self.batch_size,1,1,1).repeat(1,self.img_c,1,1)) * epsilon
    

    def _sample_t(self, batch_size, timesteps):
        return torch.randint(1, timesteps, (batch_size,))
    

    def _sample_epsilon(self, batch_size, img_h, img_w):
        return torch.normal(torch.zeros((batch_size, self.img_c, img_h, img_w)), torch.ones((batch_size, self.img_c, img_h, img_w)))
    

    def _calc_loss(self, epsilon, epsilon_pred):
        return torch.linalg.vector_norm(epsilon - epsilon_pred)
    

    def train(self):
        train_loss, test_loss = [], []
        min_test_loss = 10e10

        for epoch in range(self.num_epochs):
            epoch_train_loss = 0
            epoch_test_loss = 0
            mse = nn.MSELoss()
            with tqdm.tqdm(self.loader_train, unit="batch") as tepoch:
                for batch_idx, (data, _) in enumerate(tepoch):
                    data = data.to(f'cuda:{self.rank}')
                    epsilon = self._sample_epsilon(self.batch_size, self.img_h, self.img_w)
                    t = self._sample_t(self.batch_size, self.timesteps)
                    xt = self._calc_xt(epsilon, t, data)
                    epsilon_pred = self.model(xt, t)
                    loss = mse(epsilon, epsilon_pred)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    epoch_train_loss += loss.item() * len(data) / len(self.loader_train.dataset)
                    if batch_idx % 20 == 0 and self.rank==0:
                        tepoch.set_description(f"[Rank {self.rank}] Train Epoch {epoch}")
                        tepoch.set_postfix(loss=epoch_train_loss)
                # if self.rank == 0:
                #     print(f'[Rank {self.rank}] Epoch: {} Train Loss: {epoch_train_loss}')
                train_loss.append(epoch_train_loss)

                with tqdm.tqdm(self.loader_test, unit="batch") as tepoch:
                    for batch_idx, (data, _) in enumerate(tepoch):
                        data = data.to(f'cuda:{self.rank}')
                        with torch.no_grad():
                            epsilon = self._sample_epsilon(self.batch_size, self.img_h, self.img_w)
                            t = self._sample_t(self.batch_size, self.timesteps)
                            xt = self._calc_xt(epsilon, t, data)
                            epsilon_pred = self.model(xt, t)
                            loss = mse(epsilon, epsilon_pred)
                            epoch_test_loss += loss.item() * len(data) / len(self.loader_test.dataset)
                            if batch_idx % 20 == 0 and self.rank==0:
                                tepoch.set_description(f"[Rank {self.rank}] Train Epoch {epoch}")
                                tepoch.set_postfix(loss=epoch_test_loss)
                    if self.rank == 0:
                        print(f'[Rank {self.rank}] Epoch: {epoch} Train Loss: {epoch_train_loss} Test Loss: {epoch_test_loss}')
                    test_loss.append(epoch_test_loss)
                
                if epoch_test_loss < min_test_loss and self.rank == 0:
                    torch.save(self.model.state_dict(), '/home/kk2720/dl/diffusion-model/model/cifar_simple_diffusion1.pt')
                    min_test_loss = epoch_test_loss
        
        return train_loss, test_loss
