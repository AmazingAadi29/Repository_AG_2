# 1) How can different epochs have very different val_loss but very similar val_accuracy?

In short, loss is dependent on the probability of the prediction and accuracy is dependent ONLY on the output. Each epoch is a run through the data. Let's say for one epoch, the image number '2' is shown. The model will process it and generate probabilities about whether the number is 0-9. If the model determines the probability of the image being '2' as 0.99, the confidence would be high as the model is almost completely sure the number is 2. Because the model is very confident, there will be less loss and same accuracy. If 


Formula for binary crossentropy: $$L=-\frac{1}{N}\sum_{j=1}^{N}\left[ t_{j}log(p_{j})+(1-t_{j})(log(1-p_{j}))\right]$$
