import os, shutil

original_dir = r'/home/eirikur/deep/project/dataset'
new_dir = r'/home/eirikur/deep/project/cats_and_dogs_small'
os.mkdir(new_dir)

train_dir = os.path.join(new_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(new_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(new_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

#cats
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]

for name in fnames:
    src = os.path.join(original_dir, name)
    dst = os.path.join(train_cats_dir, name)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1250)]

for name in fnames:
    src = os.path.join(original_dir, name)
    dst = os.path.join(validation_cats_dir, name)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1250, 1500)]

for name in fnames:
    src = os.path.join(original_dir, name)
    dst = os.path.join(test_cats_dir, name)
    shutil.copyfile(src, dst)

# dogs
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]

for name in fnames:
    src = os.path.join(original_dir, name)
    dst = os.path.join(train_dogs_dir, name)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1250)]

for name in fnames:
    src = os.path.join(original_dir, name)
    dst = os.path.join(validation_dogs_dir, name)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1250, 1500)]

for name in fnames:
    src = os.path.join(original_dir, name)
    dst = os.path.join(test_dogs_dir, name)
    shutil.copyfile(src, dst)

print('total training cat images: ', len(os.listdir(train_cats_dir)))
print('total training dog images: ', len(os.listdir(train_dogs_dir)))
print('total validation cat images: ', len(os.listdir(validation_cats_dir)))
print('total validation dog images: ', len(os.listdir(validation_dogs_dir)))
print('total test cat images: ', len(os.listdir(test_cats_dir)))
print('total test dog images: ', len(os.listdir(test_dogs_dir)))
