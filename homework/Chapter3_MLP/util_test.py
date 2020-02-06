import numpy as np



def tester(x_train_data, y_train_data, model, trained_dict):

    x_train_data, y_train_data = x_train_data.cpu().numpy(), y_train_data.cpu().numpy().reshape(-1)    

    
    model.load_state_dict(trained_dict)
    
    

    test_x1 = np.linspace(-2, 2, 500)
    test_x2 = np.linspace(-2, 2, 600)
    X1, X2 = np.meshgrid(test_x1, test_x2)
    
    test_X = np.dstack((X1, X2)).reshape(-1,2)
    test_result = model(torch.tensor(test_X, dtype = torch.float, device = device))
    test_result = test_result.view(600,500).detach().cpu().numpy()
x_data = np.array([1,2,3,4])


y_data = np.array([-1,-2,-3,-4])
x,y = np.meshgrid(x_data,y_data)
print(x)
print(y)