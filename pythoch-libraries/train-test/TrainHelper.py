"""

Helper tool while training the Neural Network models.

__author__ = "Pranav Bajaj"
__copyright__ = Copyright (c) 2023 Clear Guide Medical Inc.

"""



class EarlyStopping: 
    def __init__(self, patience=1, min_delta=0): 
        self.patience = patience 
        self.min_delta = min_delta 
        self.counter = 0 
        self.min_validation_loss = float('inf')
        self.model = None 
        
    def early_stop(self, validation_loss, model): 
        if validation_loss < self.min_validation_loss: 
            self.min_validation_loss = validation_loss
            self.counter = 0 
            self.model = model  
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience: 
                return True 
        
        return False 
    
    def get_model(self,): 
        return self.model