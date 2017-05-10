#Segment analyze ml challenge 

a response to Siraj segment analyze challenge (https://www.youtube.com/watch?v=si8zZHkufRY)

I have created a neural network to predict if a games rating from its title.

I had some problems with it only output scores in the high to midrange ranges (approx. half of the available labels)

When the train.py is executed, it will train the models (word and NN) on the included ign data (from kaggle https://www.kaggle.com/egrinstein/20-years-of-games) but a trained model is already included, so feel free to skip this step

When the predict.py is executed, it will predict the ratting from the saved models, parse the game titles in as argument when the script is executed for get predictions


    python predict.py "sister fight" "test car" "race to death" "plug" "bad naibors" "nail it" "skate 2020" "overtake" "call of slams" "Purr Pals"
    
    I think sister fight is a great game!
    
    I think test car is a amazing game!
    
    I think race to death is a amazing game!
    
    I think plug is a okay game!
    
    I think bad naibors is a amazing game!
    
    I think nail it is a great game!
    
    I think skate 2020 is a amazing game!
    
    I think overtake is a amazing game!
    
    I think call of slams is a great game!
    
    I think Purr Pals is a mediocre game!
