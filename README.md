# NEURAL-STYLE-TRANSFER

*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: SHRIDHAR B MAREPPAGOL

*INTERN ID*: CT04DM1492

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

## TASK DESCRIPTION

This Python script performs Neural Style Transfer, a process where the style of one image (like a painting) is combined with the content of another image (like a photo) to generate a new, stylized image. It uses a pre-trained VGG19 convolutional neural network from the torchvision.models library to extract features from both the content and style images. These features are then used to optimize a target image that blends the content of one and the style of the other.

The code first loads and preprocesses the images, extracts deep features from certain layers of the VGG19 model, and computes two types of loss: content loss (how much the target image deviates from the content image) and style loss (how much the target image differs in texture and color style from the style image). These losses are minimized using gradient descent for 1000 iterations, resulting in a final output image that is saved to disk as a new stylized version of the original photo.

How it works: 
1.Feature Extraction Using a Pretrained CNN:
The code uses the VGG19 model, a deep convolutional neural network pretrained on ImageNet, to extract hierarchical features from both the content and style images. Different layers in VGG19 capture different types of information — early layers capture basic textures and colors (style), while deeper layers capture the actual shapes and objects (content).

2.Content and Style Representation:
   Content Features: Extracted from a deeper convolutional layer (conv4_2), representing the high-level content of the content image.
   Style Features: Extracted from multiple layers (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1), representing textures and patterns. Style features are summarized using Gram matrices, which capture correlations                       between different filter responses and effectively represent the style.

3.Optimization of the Target Image:The algorithm initializes the target image as a copy of the content image and iteratively updates it to minimize a weighted sum of two losses:
    Content Loss: The difference between content features of the target and content images.
    Style Loss: The difference between Gram matrices of the target and style images at several layers.
    Using gradient descent (Adam optimizer), the target image is updated to reduce both losses, balancing content preservation and style transfer. Over many iterations, the target image gradually adopts the style of the style image while maintaining the content structure.

4.Final Output:
   After the optimization loop completes, the resulting tensor is converted back to a viewable image and saved as the stylized output. This image visually blends the content of the original photo with the            artistic style of the chosen painting or style image.

Key Features:This neural style transfer code leverages a pretrained VGG19 convolutional neural network to separate and extract content and style features from input images. The content of an image is captured from deeper layers of the network, while style is represented through the Gram matrices computed from feature correlations across multiple layers. By optimizing a target image iteratively, the code blends the style of one image with the content of another, resulting in a stylized output that preserves the original content but adopts the artistic textures and colors of the style image.

Key features of the implementation include its use of GPU acceleration for faster processing, the ability to customize the relative importance of style versus content through adjustable weights, and the incorporation of multiple layers for richer style representation. Additionally, the code includes image preprocessing steps like resizing and normalization, as well as postprocessing to convert the optimized tensor back into a viewable image format. This combination of techniques allows for high-quality and flexible artistic style transfer on images.

Applications:Neural style transfer has a wide range of applications across various fields. In digital art and graphic design, it enables artists and designers to create unique, visually compelling artworks by blending photographic content with famous artistic styles, opening up new avenues for creative expression. It’s also used in photo editing apps to apply artistic filters that transform ordinary photos into paintings or sketches in the style of renowned artists.

Beyond art, neural style transfer finds use in advertising and marketing to produce eye-catching visuals that stand out. In entertainment and media, it can enhance visual effects and animations with distinctive aesthetics. Additionally, it has potential in augmented reality (AR) and virtual reality (VR) to create immersive environments with customized artistic styles, enriching user experiences. Researchers also explore style transfer techniques to better understand neural network representations and image processing.

This neural style transfer code uses a deep learning approach to blend the content of one image with the artistic style of another. By leveraging a pre-trained VGG19 convolutional neural network, it extracts features that represent both content and style. The algorithm iteratively optimizes a target image to minimize the difference in content with the original image while matching the style features of the style image, using losses calculated from feature maps and Gram matrices.

The process produces a new image that retains the structure and details of the content image but is rendered in the style of the chosen artwork. This technique enables creative applications in digital art, photo editing, and media production, offering a powerful tool to generate visually striking stylized images.

## OUTPUT

![Image](https://github.com/user-attachments/assets/1223be14-fd9d-4bc7-a3d4-9d1249664273)

![Image](https://github.com/user-attachments/assets/86d13532-6683-4b80-848e-3c13375d4423)















