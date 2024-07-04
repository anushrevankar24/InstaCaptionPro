
import hashlib

def image_to_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()


