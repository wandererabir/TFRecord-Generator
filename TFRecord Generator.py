import PIL
import tensorflow as tf
import hashlib
import io
import os
import untangle


if __name__ == '__main__':

    data_dir = 'dataset_path'

    tfrecord_path = 'tfrecord_path'

    writer = tf.io.TFRecordWriter(tfrecord_path)

    annotations_dir = os.path.join(data_dir, 'annotations')
    examples_list = os.listdir(annotations_dir)
    for idx, example in enumerate(examples_list):
        if example.endswith('.xml'):
            path = os.path.join(annotations_dir, example)
            xml_obj = untangle.parse(path)
            tf_example = xml_to_tf_example(xml_obj)
            writer.write(tf_example.SerializeToString())

    writer.close()

def xml_to_tf_example(xml_obj):
    label_map_dict = {'dog': 1, 'cat': 2}

    full_path = xml_obj.annotation.path.cdata
    filename = xml_obj.annotation.filename.cdata
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(xml_obj.annotation.size.width.cdata)
    height = int(xml_obj.annotation.size.height.cdata)

    xmin = []
    ymin = []
    xmax = []
    ymax = []

    classes = []
    classes_text = []
    truncated = []

    for obj in xml_obj.annotation.object:

        xmin.append(float(obj.bndbox.xmin.cdata) / width)
        ymin.append(float(obj.bndbox.ymin.cdata) / height)
        xmax.append(float(obj.bndbox.xmax.cdata) / width)
        ymax.append(float(obj.bndbox.ymax.cdata) / height)
        classes_text.append(obj.name.cdata.encode('utf8'))
        classes.append(label_map_dict[obj.name.cdata])
        truncated.append(int(obj.truncated.cdata))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/source_id': bytes_feature(filename.encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
        'image/object/truncated': int64_list_feature(truncated),
    }))
    return example