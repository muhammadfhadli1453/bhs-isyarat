

import imp
from flask import Flask, render_template, Response
import cv2
import os
from gtts import gTTS
import playsound
import numpy as np
from warnings import simplefilter 
from pydub import AudioSegment
from pydub.playback import play
import threading

simplefilter(action='ignore', category=FutureWarning)

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
np.seterr(all="ignore")
import tensorflow as tf


app=Flask(__name__)

def gen_frames():  
    def predict(image_data):

        predictions = sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        max_score = 0.0
        res = ''
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if score > max_score:
                max_score = score
                res = human_string
        return res, max_score

    def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

        return text_size

    label_lines = [line.rstrip() for line
                    in tf.gfile.GFile("logs/trained_labels.txt")]

    with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    camera = cv2.VideoCapture(0)

    if camera.isOpened():
        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            c = 0

            res, score = '', 0.0
            i = 0
            mem = ''
            consecutive = 0
            sequence = ''
            text = ''
            
            while True:
                ret, img = camera.read()
                img = cv2.flip(img, 1)
                
                if ret:
                    x1, y1, x2, y2 = 100, 100, 300, 300
                    img_cropped = img[y1:y2, x1:x2]

                    c += 1
                    image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
                    
                    a = cv2.waitKey(1) # waits to see if `esc` is pressed
                    
                    if i == 4:
                        res_tmp, score = predict(image_data)
                        res = res_tmp
                        i = 0
                        if mem == res:
                            consecutive += 1
                        else:
                            consecutive = 0
                        if consecutive == 1 and res not in ['nothing']:
                            if res == 'space':
                                if sequence[-1] == ' ':
                                    pass
                                else:
                                    sequence += ' '
                                    text += ' '
                            elif res == 'del':
                                sequence = sequence[:-1]
                                text = text[:-1]
                            elif len(sequence) > 0:
                                if sequence[-1] == res:
                                    pass
                                elif score < 0.2:
                                    sequence += ''
                                else:
                                    sequence += res
                                    text += res
                            else:
                                sequence += res
                                text += res
                            consecutive = 0
                        if res in ['nothing']:
                            if text:
                                file = gTTS(text=text, lang="id")
                                file.save("tmp.mp3")
                                text = ''
                                # playsound.playsound('tmp.mp3', True)
                                # os.remove('tmp.mp3')
                                sound = AudioSegment.from_mp3('tmp.mp3')
                                t = threading.Thread(target=play, args=(sound,))
                                t.start()
                    i += 1
                    cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
                    # cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                    mem = res
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
                    # cv2.imshow("img", img)
                    # cv2.putText(img, '%s' % (sequence.upper()), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    
                    draw_text(img, '%s' % (sequence.upper()), font_scale=2, pos=(100,410), text_color_bg=(255, 0, 0))
                    if len(sequence) == 20:
                        sequence = ''
                    ret, buffer = cv2.imencode('.jpg', img)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    # img_sequence = np.zeros((200,1200,3), np.uint8)
                    # cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    # cv2.imshow('sequence', img_sequence)
    else:
        img = cv2.imread("templates/kamera.png")
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # while True:
    #     success, frame = camera.read()  # read the camera frame
    #     if not success:
    #         break
    #     else:
    #         detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #         faces=detector.detectMultiScale(frame,1.1,7)
    #          #Draw the rectangle around each face
    #         for (x, y, w, h) in faces:
    #             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    #         ret, buffer = cv2.imencode('.jpg', frame)
    #         frame = buffer.tobytes()
    #         yield (b'--frame\r\n'
    #                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# @app.route('/get_sound')
# def get_sound():
#     if 

if __name__=='__main__':
    app.run(debug=True)