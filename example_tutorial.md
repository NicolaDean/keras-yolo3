# HOW TO LOAD YOLO:
You can find the code of this example in [yolo_no_args.py](yolo_no_args.py)
## Step1:
### Define all the path to required files.
- **Annotation File**: Contain a list of images in the format [filename] [boxes information]
- **Classes File**: Contain on each row the name of a class
- **Anchor File**: Define the starting anchor (use the default file)
- **Model Weights**: Path to the model .h5 file

```python
annotation_path = './test/_annotations.txt'
classes_path    = './test/_classes.txt'
anchors_path    = 'model_data/yolo_anchors.txt'
model_weights   = 'yolo_boats_final.h5'
```

## Step2:
### Define some usefull variables to store YOLO configuration info
```python
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors     = get_anchors(anchors_path)
input_shape = (416,416) # multiple of 32, hw
```

## Step3:
### Load the annotation file lines in a List of lines

```python
with open(annotation_path) as f:
        lines = f.readlines()
```

## Step4:
### Now create data generator
 ```python
gen = data_generator_wrapper('./test/',lines, 1, input_shape, anchors, num_classes)

 ```

## NOTE: Step 3,4 can be customized with your own data pipeline

## Step5:
### Since owner of the repository made YOLO class as "cmd args compatible" we need to convert args => file constants.
```python
class args:
    def __init__ (self, model_path   = 'yolo_boats_final.h5',
                        anchors_path = annotation_path,
                        classes_path = classes_path,
                        score = 0.3,
                        iou = 0.45,
                        model_image_size = (416, 416),
                        gpu_num = 1):
        self.model_path  = model_path
        self.anchors_path  = anchors_path
        self.classes_path  = classes_path
        self.score  = score
        self.iou  = iou
        self.model_image_size  = model_image_size
        self.gpu_num  = gpu_num

argss = args()

yolo = YOLO(**vars(argss))
```

Probably there are some cleaner way to do so, and you are free to use them.

## Step6:
### Now We can Run Yolo on inference!

```python
    while True:
        print("loading Data")
        dataset = next(gen)
        data   = dataset[0][0]
        boxes  = dataset[2]
        #Yolo has 3 head at different step of the network
        labels_1 = dataset[0][1] #First  Yolo head
        labels_2 = dataset[0][2] #Second Yolo head
        labels_3 = dataset[0][3] #Third  Yolo head

        from PIL import Image, ImageFont, ImageDraw
        
        #Convert image in int8
        img = np.uint8(data[0]*255)
        #Convert it into PIL IMAGE to show it to screen
        img = Image.fromarray(img)
        #Inference
        r_image = yolo.detect_image(img)
        #Show the result
        r_image[0].show()
        ex = input('press a key')
        if ex == 'q':
            exit()
```