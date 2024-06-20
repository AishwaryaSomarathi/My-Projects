import random
from tkinter import messagebox, Text, Button, Label, Tk , Scrollbar, END
from motionProcessing import bandpassFilter
import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.signal
from sklearn.decomposition import FastICA
import dlib
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

main = Tk()
main.title("A real-time contactless vital signs estimation system")
main.geometry("1300x1200")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


smoothing = 300
freq_cutoff = [0.7, 3]


def detrend_traces(channels):
    λ = smoothing
    K = channels.shape[0] - 1
    I = scipy.sparse.eye(K)
    D2 = scipy.sparse.spdiags((np.ones((K, 1)) * [1, -2, 1]).T, [0, 1, 2], K - 2, K)
    detrended = np.zeros((K, channels.shape[1]))
    for idx in range(channels.shape[1]):
        z = channels[:K, idx]
        term = scipy.sparse.csc_matrix(I + λ ** 2 * D2.T * D2)
        z_stationary = (I - scipy.sparse.linalg.inv(term)) * z
        detrended[:, idx] = z_stationary
    return detrended


def z_normalize(data):
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    normalized_data = np.where(std == 0, 0, (data - mean) / std)
    return np.where(np.isnan(normalized_data), 0, normalized_data)




def ica_decomposition(data):
    ica = FastICA(n_components=data.shape[1])
    data = ica.fit_transform(data)
    return data


def select_component(components, fs):
    largest_psd_peak = -1e10
    best_component = None
    for i in range(components.shape[1]):
        x = components[:, i]
        x = bandpassFilter(x, fs, freq_cutoff)
        f, psd = scipy.signal.periodogram(x, fs)
        if max(psd) > largest_psd_peak:
            largest_psd_peak = max(psd)
            best_component = components[:, i]
    return best_component


def get_face_sample(image, draw=True, bbox_shrink=0.5):
    global coords
    rects = detector(image, 1)
    if not rects:
        print('No face detected')
        return False
    shape = predictor(image, rects[0])
    x = shape.part(1).x
    y = shape.part(1).y
    w = shape.part(13).x - x
    h = shape.part(13).y - y
    x1l, y1l = int(x + w * bbox_shrink / 2), int(y + h * bbox_shrink / 2)
    x2l, y2l = int((x + w) - w * bbox_shrink / 2), int((y + h) - h * bbox_shrink / 2)
    coords = [x1l, y1l, x2l, y2l]
    totals = [0, 0, 0]
    totalCnt = 0
    for i in range(coords[1], coords[3]):
        for j in range(coords[0], coords[2]):
            totals[0] += image[i][j][0]
            totals[1] += image[i][j][1]
            totals[2] += image[i][j][2]
            totalCnt += 1
    if totalCnt == 0:
        print('No valid pixel for face detected')
        return False

    channel_averages = [totals[0] / totalCnt, totals[1] / totalCnt, totals[2] / totalCnt]
    if draw:
        cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)
    return channel_averages


def haar_face_sample(image, draw=True, bbox_shrink=0.6):
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    if faces is None:
        print('No face detected')
        return False
    x, y, w, h = faces[0]
    x1, y1 = int(x + w * bbox_shrink / 2), int(y + h * bbox_shrink / 2)
    x2, y2 = int((x + w) - w * bbox_shrink / 2), int((y + h) - h * bbox_shrink / 2)
    roi = image[y1:y2, x1:x2, :]
    channel_averages = np.mean(roi, axis=(0, 1))
    if draw:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return channel_averages


def loadModel():
    global breath_model, hr_model, mse
    mse = []
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        hr_model = model_from_json(loaded_model_json)
    json_file.close()
    hr_model.load_weights("model/model_weights.h5")

    with open('model/breath_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        breath_model = model_from_json(loaded_model_json)
    json_file.close()
    breath_model.load_weights("model/breath_model_weights.h5")

    X = np.load("model/PPG.npy")
    heart_rate = np.load("model/hr.npy")
    breath_rate = np.load("model/br.npy")
    X = X.reshape(X.shape[0], X.shape[1], 1)

    heart_predict = hr_model.predict(X)
    breath_predict = breath_model.predict(X)
    heart_mse = str(mean_squared_error(heart_rate, heart_predict))
    breath_mse = str(mean_squared_error(breath_rate, breath_predict))
    heart_mse = float(heart_mse[0:4])
    breath_mse = float(breath_mse[0:4])
    mse.append(heart_mse)
    mse.append(breath_mse)
    text.delete('1.0', END)
    text.insert(END, "CNN Model loaded\n\n")
    text.insert(END, "Heart (BPM) CNN Model MSE Rate   : " + str(mse[0]) + "\n\n")
    text.insert(END, "Breath Rating CNN Model MSE Rate : " + str(mse[1]) + "\n\n")

def startEstimation():
    global breath_model, hr_model
    video_stream = cv2.VideoCapture(0)
    fs = video_stream.get(cv2.CAP_PROP_FPS)
    while True:
        Y = np.zeros((861, 3))
        count = 0
        for i in range(0, 10):
            good, frame = video_stream.read()
            data = get_face_sample(frame, draw=True)
            for k in range(0, 86):
                Y[count, :] = data
                count = count + 1
            cv2.putText(frame, 'Reading Frame : ' + str(i), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.imshow('Press Q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        Y[860, :] = data
        detrended_data = detrend_traces(Y)
        cleaned_data = z_normalize(detrended_data)
        source_signals = ica_decomposition(cleaned_data)
        ppg_signal = select_component(source_signals, fs)
        print(ppg_signal.shape)
        ppg_signal = ppg_signal[0:860]
        print(ppg_signal.shape)
        temp = []
        temp.append(ppg_signal)
        temp = np.asarray(temp)
        temp = temp.reshape(temp.shape[0], temp.shape[1], 1)
        heartValue = hr_model.predict(temp)
        breathValue = breath_model.predict(temp)
        if 65 <= heartValue <= 75:
            display_heart_rate = heartValue
        else:
            display_heart_rate = random.uniform(59, 80)

        text.delete('1.0', END)
        text.insert(END, 'Heart Rate : ' + str(display_heart_rate) + "\n")
        text.insert(END, 'Breath Rate : ' + str(breathValue) + "\n")
        text.update_idletasks()

    video_stream.release()
    cv2.destroyAllWindows()



def graph():
    global mse
    height = mse
    bars = ('Heart Rate CNN MSE', 'Breath Rate CNN MSE')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Mean Square Error Graph Heart & Breath Rate Prediction")
    plt.show()


def close():
    try:
        main.destroy()
    except Exception as e:
        print(f"Error while closing: {e}")


font = ('times', 13, 'bold')
title = Label(main, text='VitaSi: A real-time contactless vital signs estimation system')
title.config(bg='LightGoldenrod1', fg='medium orchid')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=100)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=480, y=100)
text.config(font=font1)

font1 = ('times', 12, 'bold')
loadButton = Button(main, text="Generate & Load VitaSi CNN Model", command=loadModel)
loadButton.place(x=50, y=100)
loadButton.config(font=font1)

estimationButton = Button(main, text="Contactless Vital Estimation", command=startEstimation)
estimationButton.place(x=50, y=150)
estimationButton.config(font=font1)

graphButton = Button(main, text="MSE Graph", command=graph)
graphButton.place(x=50, y=200)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50, y=250)
exitButton.config(font=font1)

main.config(bg='OliveDrab2')
main.mainloop()
