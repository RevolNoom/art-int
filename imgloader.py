import numpy

def load_data(images_filename, labels_filename):
    """ 
        SYNOPSIS: (for MNIST dataset files only)
            load_data(images_filename, labels_filename)

        Return array "data" of tuples with 2 entries:
            data[x][0]: 
                +) A 28x28 image in form of an array with 784 integers
                +) Oriented row-wise, which mean, data[x][0] to data[x][rows-1] are of the first row
            data[x][1]:
                +) Label of image x
                +) An array with length 10
                +) If data[x][1][i] = 1.0, then data[x] is digit "i"
                +) Otherwise, data[x][1][i] = 0.0
    """

    # "rb" = Read file in Binary mode
    images_file = open(images_filename, "rb")
    labels_file = open(labels_filename, "rb")

    # Testing magic number
    magic_number = (int.from_bytes(labels_file.read(4), "big", signed = False), int.from_bytes(images_file.read(4), "big", signed = False))
    
    # These debugging lines is written when I wasn't aware of int.from_bytes()
    # Ignore them
    # if magic_number != (0x00000801, 0x00000803):
    #    print("You magically get (" + str(magic_number[0]) + ", " + str(magic_number[1]) + ") instead of (2049, 2051).")
    #    print("Maybe your computer processor is from Intel?")
    #    return None

    # Number of things we are getting:
    number_of_labels= int.from_bytes(labels_file.read(4), "big", signed = False)
    number_of_imgs  = int.from_bytes(images_file.read(4), "big", signed = False)

    if number_of_labels != number_of_imgs:
        print("The amounts of labels and images are not equal!")
        print("Labels: " + str(number_of_labels))
        print("Images: " + str(number_of_images))
        return []

    # Getting images size:
    rows = int.from_bytes(images_file.read(4), "big", signed = False)
    cols = int.from_bytes(images_file.read(4), "big", signed = False)

    # Output results here:
    data = []
    
    # Getting the images and label
    for i in range(number_of_imgs):
        image = [int.from_bytes(images_file.read(1), "big", signed = False) for i in range(rows*cols)]
        temp_label = int.from_bytes(labels_file.read(1), "big", signed = False)
        if temp_label not in range(10):
            print("Woa woa hey hey is " + str(temp_label) + " a new kind of digit?")
            return []
        # Label is an array with 10 slots, each corresponding to one of 0 - 9
        label = [0.0 for i in range(10)]
        label[temp_label] = 1.0

        data.append((image, label))

    return data

# This line is for debugging purpose
# load_data("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
