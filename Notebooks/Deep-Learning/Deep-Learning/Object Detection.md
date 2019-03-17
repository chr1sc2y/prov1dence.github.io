# transferObject Detection

## Object Localization

- figure out where in the picture is the car
- detect all objects and localize them

### Bounding Boxes

- $p_c​$: is there any object
- $b_x$: x of midpoint
- $b_y$: y of midpoint
- $b_h$: the height
- $b_w$: the width
- class label

### sliding windows detection



1. 

- input a small rectangular region
- make predictiosn in a ConvNet in the region
- slide the square across the entire image

2. 

- convolution on the entire image
- Make predictions at the same time



## YOLO algorithm

- single convolutional implementation
- take the midpoint of each of the objects
- assign the object to the grid cell containing the midpoint
- output labels

### Labels

#### Single Label

- precise bounding boxes
- $ y = \begin{bmatrix} p_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ c_2 \\ c_3 \\ ... \end{bmatrix} $
  - (5 + m) dimensional output
    - m: number of class of objects
  - $ 0 \leq b_x, b_y \leq 1 $
- $ y = [0 ...]^T ​$

#### Target Label

- $ n * n * y $
  - n: number of cells

### Intersection over Union (交并比)

- the intersection over union of two bounding boxes
- correct if IoU $ \geq $ 0.5 (at least)

### Non-max Suppression (非极大值抑制)

- clean up multiple detections
  - may end up with multiple detections on each object

#### Process

- discard all cells with low $p_c$ under threshold
- pick the cells with the highest $p_c$
- suppress all other cells with a high overlap with a high IoU

### Anchor Boxes

- detect overlapping objects
- assign an object
  - to the grid cell containing the midpoint
  - to an anchor box with the highest IoU
- $ y = \begin{bmatrix} p_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ c_2 \\ c_3 \\ ... \\ p_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ c_2 \\ c_3 \\ ... \end{bmatrix} $
  - (5 + m) * a dimensional output



## Face Recognition

| Face Verification                           | Face Recognition                        |
| ------------------------------------------- | --------------------------------------- |
| classify whether an input image is a person | classify who the person on the image is |

### Face Verification

#### similarity function

- d(img1, img2) = degree of difference between images
  - $d \leq \tau $: same person
  - $d > \tau $: different person
  - $ \tau ​$: hyperparameter

#### Siamese Network

- learning parameters that
  - if $x^i, x^j​$ are the same person, then $ ||f(x^i) - f(x^j) || ​$ is small
  - if $x^i, x^j$ are not the same person, then $ ||f(x^i) - f(x^j) || $ is large

#### Triplet Loss

- triplet
  - Anchor
  - Positive
  - Negative
- minimize $ ||f(A) - f(P)||^2 \leq ||f(A) - f(N)||^2 $
  - d(A, P) $\leq$ d(A ,N) 
  -  $ ||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + \alpha \leq 0 ​$
    - $\alpha$: margin

##### Loss Function

- $ loss(A, P, N) = max(||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + \alpha, 0) $

- $ J = \sum_{i=1}{m} loss(A^{(i)}, P^{(i)}, N^{(i)}) $

##### Triplets

- choose triplets that are hard to train
  - d(A, P) $\approx​$ d(A ,N)

#### Binary Classification





## Neural Style Transfer

### Cost Function

$ J(G) = \alpha J_{content}(C, G) + \beta J_{style}(S, G) ​$

#### Content Cost Function

1. use pre-trained ConvNet
2. let $a^{[l](C)}$ and $a^{[l](G)}$ be the activation of layer l
3. if $a^{[l](C)}​$ and $a^{[l](G)}​$ are similar, both images have similar content

- $ J_{content}(C, G) = \frac{1}{2} || a^{[l](C)} - a^{[l](G)} ||^2 ​$

#### Style Cost Function

##### Style Matrix for the Style Image

- $a_{ijk}^{[l]}$: activation a (i, j, k)

- $G_{kk'}^{[l](S)} = \sum_i^{n_h^{[l]}} \sum_j^{n_w^{[l]}} a_{ijk}^{[l](S)} a_{ijk'}^{[l](S)}​$
  - $G^{[l](S)}​$ size: $n_c^{[l]} * n_c^{[l]}​$
  - S: style image
  - $kk'​$: measure how correlated are channels $k​$ and $k'​$

##### Style Matrix for the Generated Image

$G_{kk'}^{[l](G)} = \sum_i^{n_h^{[l]}} \sum_j^{n_w^{[l]}} a_{ijk}^{[l](G)} a_{ijk'}^{[l](G)}​$

##### Cost Function

$ J_{style}(S, G) = || G^{[l](S)} - G^{[l](G)} ||^2 = \frac{1}{(2n_h^{l}n_w^{l}n_c^{l})^2} \sum_k \sum_{k'} (G_{kk'}^{[l]()} - G_{kk'}^{[l](G)})^2$

### Generate Image

1. Initiate G randomly
2. use gradient descent to minimize J(G)

