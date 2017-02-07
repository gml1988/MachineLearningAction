from __future__ import print_function
warned_of_error = False

def create_cloud(oname, words, maxsize=120, fontname='Lobster'):
    try:
        from pytagcloud import create_tag_image, make_tags
    except ImportError:
        if not warned_of_error:
            print("Could not import pytagcloud. Skipping cloud generation.")
        return

    words = [(w, int(v * 10000)) for v, w in words]
    tags = make_tags(words, maxsize=maxsize)
    create_tag_image(tags, oname, size=(1800, 1200), fontname=fontname)