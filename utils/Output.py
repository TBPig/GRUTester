class Output:
    def __init__(self):
        self.train_loss = []
        self.train_idx = []
        self.consume_time = 0

        self.test_loss = []
        self.test_idx = []
        self.test_cr = []

        self.model_name = ""
        self.model_info = ""
        self.test_info = {}

    def add_train_info(self, loss, x):
        self.train_loss.append(loss)
        self.train_idx.append(x)
        return self

    def add_test_info(self, cr, loss, x):
        self.test_cr.append(cr)
        self.test_loss.append(loss)
        self.test_idx.append(x)
        return self

    def set_time(self, time):
        self.consume_time = time
        return self

    def add_info(self, info_name, info):
        self.test_info[info_name] = info
        return self

    def get_model_info(self):
        return self.model_info

    def set_model(self, model):
        self.model_info = model.get_info()
        self.model_name = model.name
        return self
