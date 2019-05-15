def plain_log(filename, text):
    fp = open(filename,'a')
    fp.write(text)
    fp.close()
