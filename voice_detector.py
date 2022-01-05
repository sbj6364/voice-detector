# 화자 데이터(F1.wav, F2.wav ..)
# 테스트 데이터(1.wav, 2.wav ..)




import numpy as np
import librosa
from sklearn.mixture import GaussianMixture

sr = 16000


#------------------------
# train data loading
#------------------------

train_voice = []
train_mfcc = []
train_files = ["F1.wav", "F2.wav", "M1.wav", "M2.wav"] # 화자 음성 파일 이름

for i in range(len(train_files)):
    y, sr = librosa.load(train_files[i], sr=sr) # 화자데이터 경로
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=24, n_fft=512).T
    train_voice.append(y)
    train_mfcc.append(mfcc)



#------------------------
# test data loading
#------------------------

test_voice = []
test_mfcc = []
test_files = ["1.wav", "2.wav", "3.wav", "4.wav", "5.wav"] # 테스트 파일 이름

for i in range(len(test_files)):
    y, sr = librosa.load(test_files[i], sr=sr) # 테스트 파일 경로
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=24, n_fft=512).T
    test_voice.append(y)
    test_mfcc.append(mfcc)



#------------------------
# GMM model training
#------------------------

print("\n\n< 모델 학습 >\n")

gmms = []
speakers = ['F1', 'F2', 'M1', 'M2']

for i in range(4):
  print('{} 음성 학습중..'.format(speakers[i]))
  gmm = GaussianMixture(n_components=5, covariance_type='tied', reg_covar=1e-1, random_state=1)
   
  mfcc = train_mfcc[i]
  
  gmm.fit(mfcc)
  gmms.append(gmm)

print("\n학습 완료!")



#------------------------
# Test
#------------------------

print("\n\n\n< test 화자 인식 결과 >\n")

for i in range(5):
    scores = []
    
    for j in range(4):
        scores.append(gmms[j].score(test_mfcc[i]))

    max_index = scores.index(max(scores))
    predict = speakers[max_index]
    print("test{}: ".format(i + 1) + predict)














  
