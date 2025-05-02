# Quantization Fundamentals with Hugging Face (Deeplearning.ai course)
Ref: https://learn.deeplearning.ai/courses/quantization-fundamentals

## Lesson 3: Loading ML Models with Different Data Type 

Watch `Loading models by data type`, and then answer these questions: 

### Model casting 
1. What does the `print_param_dtype()` function do? 

   *Prints the data type of the model's paramaters.*
   
2. Show the code that can be used to **cast** the model parameters to bfloat16 data type. 

   *model_bf16 = model_bf16.to(torch.bfloat16)*

3. What do we mean by performing an **inference** of a model. 

   *Using a trained machine learning or deep learning model to make predictions based on new, unseen input data.*

4. What is `deepcopy` used in this video?

   *Deepcopy is a method that make a copy of a model. It does not impact the original model in any way when changes are made to this copy.*

5. The `model_bf16` uses lesser precision than the FP32 `model`. In the video, how do they compare the impact of this low precision on the model performance? Is the performance degradation significant?

   *They compute the difference between the errors (logits) of both models. When you switch to BF16 from FP32, it does not lead to a huge performance degradation.*

### Loading model in low precision 

6. Name the Pytorch method that can be used to get the memory footprint of a model. 

   *get_memory_footprint()*

7. How is loading the model at low precision better than downcasting the weights of a loaded model?

   *It's better to load the model at low precision because it improves performance, and is more efficient in terms of memory usage and computations.*

### Linear quantisation

8. How is linear quantisation different from lower precision floating point representation?

   *Linear quantization uses integer data types and maps floats on a scale of -128 to 127, where the most negative number is turned into -128 and the most positive is 127 and the rest of the numbers are mapped linearly as integers with fewer bits. Floating point uses floats and reduces the number of bits used to store the values.*
   
9. How does linear quantization lead to lower memory footprint?

   *Linear quantization reduces memory footprint by mapping continuous floating-point values to integers with fewer bits, reducing the memory needed to store the data.*


   
