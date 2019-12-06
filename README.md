This folder contains the following Python scripts:  
    1] train.py . 
    2] test.py . 
    3] loadddata.py . 

------------------------------Instructions---------------------------------  
** The test data files should be copied to the project folder ** . 
** The test data should be a .pkl file .   


------------------------------ train.py -----------------------------------  
** This trains the neural network. It takes in one command line argument 
   which selects which model to train for, 'AB' for training for a & b
   and 'All' to train for all 8 alphabets .  
** It generates 'networkAB.pickle' or 'networkAll.pickle', depending upon
   what the model is trained for . 
** The paramters that the neural network is trained on like solver, alpha, hidden layers
   momentum, learning rate and activation function can be configured in the MLPClassifier
   function to run different experimental designs . 
** The model generated from this has been trained on balanced_data.npy, which is the 
   cleaned up version of train_data.pkl. If you wish to train the model on your own,
   replace 'balanced_data.npy' in train.py with your filename(.npy) . 
** For AB, k value for Cross Validation = 5 . 
** For All, k value for Cross Validation  = 20 . 


------------------------------ test.py -----------------------------------  
** This loads the trained model and tests on a test data set . 
** This takes in 3 command like arguments . 
            1] Type of network to test ('AB' or 'All') . 
            2] Test data file-name(.pkl file) . 
            3] Output file to save the predicted labels . 
** It will generate an output file(sys.argv[3]) that contains the predicted labels . 
** It also returns the predicted values . 


-------------------------- Trained Models --------------------------------  
** networkAB.pickle is the trained network to classify 'AB' . 
** networkAll.pickle is the trained network to classify 'All' . 
** the networkAll.pickle could not be uploaded on Github as the file size limit exceeded . 
   We have attached a Google drive link to the file here . 
   https://drive.google.com/file/d/1ebInui8Dib0ynMCfeV_cbIliAfMp0WC1/view?usp=sharing . 


------------------------- Commands to execute ----------------------------  
To train for AB model :   python train.py AB . 
To train for All model:   python train.py All . 

To test for AB model  :   python test.py AB 'test file name(.pkl)' output_AB.npy . 
To test for All model :   python test.py All 'test file name(.pkl)' output_All.npy . 

