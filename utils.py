import torch

class SaveBestModel:
    def __init__(self, path, metric, mode="max"):
        self.model = None
        self.path = path
        self.metric = metric
        self.mode = mode
        self.best_metric = None
        self.best_epoch = None
        self.best_model = None

    def __call__(self, model,epoch, optimizer,criterion, metric):
        if self.best_metric is None:
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_optimizer = optimizer.state_dict()
            self.best_criterion = criterion.state_dict()
            self.best_model = model.state_dict()
            torch.save(
                {
                    'epoch':epoch,
                    'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'criterion':criterion.state_dict(),
                },self.path
            )
        else:
            if self.mode == "max":
                if metric > self.best_metric:
                    self.best_metric = metric
                    self.best_epoch = epoch
                    self.best_model = self.model.state_dict()
                    self.best_optimizer = optimizer.state_dict()
                    self.best_criterion = criterion.state_dict()
                    torch.save(
                    {
                        'epoch':epoch,
                        'model':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'criterion':criterion.state_dict(),
                    },self.path
                    )
            else:
                if metric < self.best_metric:
                    self.best_metric = metric
                    self.best_epoch = epoch
                    self.best_model = model.state_dict()
                    self.best_optimizer = optimizer.state_dict()
                    self.best_criterion = criterion.state_dict()
                    torch.save(
                    {
                        'epoch':epoch,
                        'model':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'criterion':criterion.state_dict(),
                    },self.path
                    )

    def load_best_model(self):
        self.model.load_state_dict(self.best_model.state_dict())

    def get_best_metric(self):
        return self.best_metric

    def get_best_epoch(self):
        return self.best_epoch
    

def save_model(path, model, optimizer, criterion, epoch):
    print(f"Saving model...........")
    torch.save(
        {
            'epoch':epoch,
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'criterion':criterion.state_dict(),
        },path
    )