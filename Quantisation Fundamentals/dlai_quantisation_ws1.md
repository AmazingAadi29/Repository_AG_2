# Quantization Fundamentals with Hugging Face (Deeplearning.ai course)
Ref: https://learn.deeplearning.ai/courses/quantization-fundamentals

## Lesson 1: Introduction 

Watch `Introduction`, and then answer these questions: 
1. Andrew mentions **PyTorch** library. What is PyTorch? Which library that you have used so far in the course is similar to PyTorch?

   *TensorFlow*
   
2. Andrew also talks about **data types**. What are data types used for? List the data types that are mentioned in this video.

   *Defines the type of data a variable can hold. The data tyepes mentioned are int8, float16, bfloat16, float32 (32 bit floating point number?).*
   
3. What is the purpose of this course?

   *To teach about quantization in AI models, specifically very large generative models.*

Watch `Handling Big Models`, and then answer these questions: 

4. Why is handling big models a problem for the AI community?

   *These models aren't efficiently run except when using memory-heavy hardware, causes performance degradation.*
   
5. To solve this problem, the video mentions 2 techniques (not covered in detail) apart from quantization. What are they?

   *1) Pruning* $$\longrightarrow$$ *Removing layers in a model that do not have much importance in the model's decisions. It removes the layers based on different metrics, like the magnitudes of weights, for example.*

   *2) Knowledge Distillation* $$\longrightarrow$$ *You train a student model (output from teacher model - so a compressed model from a larger model) which helps distill knowledge easier than with just the large model.*
   
6. What makes a model "big"? Should we be concerned about where we store the model or how we run the model? Or both?

   *A large model has a large amount of paramaters. It requires lots of memory to store them, so the model should be stored in hardware that can handle it's size and it needs to be run efficiently.*
   
7. The video describes how quantization is done by moving from floating point (FP32) to integer (INT8). Explain how this makes the model "smaller".

   *Quantizaion represents model weights in a lower precision.*

   *FP32* $$\longrightarrow$$ *4 bytes per parameter, 4 times 8-bit precision, so 36 bytes (4x8 + 4)*

   *INT8* $$\longrightarrow$$ *If the matrix is quantized to 8-bit preciison (INT8), each weight would require just one byte, so a total of 9 (1x8 + 1) bytes is needed, making a smaller model.*
   
## Lesson 2: Data types & sizes 

Watch `Data types & sizes` video to answer the questions below: 

### Integer data type

The video shows the range of values for the 2 types of integer data types: **unsigned integer** and **signed integer**. 

8. Show the formula used to calculate the **range**.

   *unsigned integer* $$\longrightarrow [0, 2^{n}-1]$$

   *signed integer* $$\longrightarrow [-2^{n-1}, 2^{n-1}-1]$$
   
9. Compute the range for **unsigned int** with 8-bits. Show the steps in your calculation.

   $$[0, 2^{n}-1]; n=8 \longrightarrow [0, 2^{8}-1] = [0, 255]$$
   
10. Compute the range for **signed int** with 8-bits. Show the steps in your calculation.

   $$[-2^{n-1}, 2^{n-1}-1]; n=8 \longrightarrow [-2^{8-1}, 2^{8-1}-1] = [-128, 127]$$
   
11. Represent the values, 2, 6, 33 and 100 in the 8-bit unsigned int data format.

    $$2 \longrightarrow 00000010$$
    
    $$6 \longrightarrow 00000110$$
    
    $$33 \longrightarrow 00100001$$
    
    $$100 \longrightarrow 01100100$$
   
12. What is the value represented by **11010101** in unsigned integer format? What is the value in signed integer format?

    *unsigned integer* $$\longrightarrow 2^{7} + 2^{6} + 2^{4} + 2^{2} + 2^{0} = 128 + 64 + 16 + 4 + 1 = 213$$

    *signed integer* $$\longrightarrow -2^{7} + 2^{6} + 2^{4} + 2^{2} + 2^{0} = -128 + 64 + 16 + 4 + 1 = -43$$

### Floating point data type

13. What type of data is stored using the floating point data type?

    *Precise numbers*
    
14. Explain the 3 components in the floating point data type. Identify the components that represent the **range** and **precision**.

    *1) Sign* $$\longrightarrow \pm$$ *(Always 1 bit)*
    
    *2) Exponent* **(range)** $$\longrightarrow$$ *Impacts the range of the number*
    
    *3) Fraction* **(precision)** $$\longrightarrow$$ *Impacts the precision of the number* 
    
16. What is "floating" in this data type? Explain with an example. 

    *The fraction (precision)* $$\longrightarrow$$ *0.4999 vs 0.5, 0.4999 has a greater precision compared to 0.5
    
17. Compare FP32, FP16 and BF16 formats in terms of precision and range.

    *1) FP32* $$\longrightarrow *Most precise (23 bits), largest range (8 bits)*
    *2) FP16* $$\longrightarrow *Middle precise (10 bits), smallest range (5 bits)*
    *3) BF16* $$\longrightarrow *Least precise (7 bits), largest range (8 bits)*
    
18. The video mentions **tensor**. What is a tensor?

    *A multidimensional array that can be represented as scalars (0D), vectors (1D), or matrices (2D) [there are 3D+ tensors] in machine learning. 1D is used in deep learning.*
   
29. What PyTorch function can I use to print the range of the floating point data type? Show the code for checking the range for the brain float BF16 data type.

    *finfo* $$\longrightarrow$$ *torch.finfo(torch.bfloat16)*
    
### Impact of downcasting 

20. What is **downcasting**? Why do it? 

    *Downcasting is when we convert a higher data type, such as a float, to a lower data type, like an integer. This will induce a loss of data, making the model be **more computationally efficient** as the data is less precise.*

21. What is the impact of **downcasting** on the result of matrix multiplication?
    
    *A loss of precision.*

22. Why did the video choose matrix multiplication to check the impact of downcasting on the precision of the result?
    
    *Matric multiplication is commonly used in neural networks as you have to multiply many inputs with many weights.*

23. One of the advantages of downcasting is reduced memory footprint. How does this allow us to enable larger batch sizes? Why is having larger batch sizes beneficial to training?
   
    *Less memory is used, so you can store larger batches into memory. Larger batch sizes is beneficial because it results in more efficient training, less noise and a smoother gradient.*    
    
24. The video mentions a use case of **mixed precision training** with downcasting. Why would this work?
   
    *It works because it uses lower precision (for computations) and higher precision (to store and update weights) to make the model more efficient and accurate.*
    


   
