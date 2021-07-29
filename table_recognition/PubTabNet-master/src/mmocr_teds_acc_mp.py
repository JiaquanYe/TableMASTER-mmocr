import os
import json
import time
import pickle
from metric import TEDS
from multiprocessing import Pool

def htmlPostProcess(text):
    text = '<html><body><table>' + text + '</table></body></html>'
    return text

def singleEvaluation(teds, file_name, context, gt_context):
    # save problem log
    # save_folder = ''

    # html format process
    htmlContext = htmlPostProcess(context)
    htmlGtContext = htmlPostProcess(gt_context)
    # Evaluate
    score = teds.evaluate(htmlContext, htmlGtContext)

    print("FILENAME : {}".format(file_name))
    print("SCORE    : {}".format(score))
    return score


if __name__ == "__main__":

    t_start = time.time()
    pool = Pool(64)
    start_time = time.time()
    predFile = './predResult.pkl'
    gtJsonFile = './gtVal.json'
    problemFile = './problem.txt'
    fid = open(problemFile, 'w')

    threshold = 0.80
    problemList = []

    # Initialize TEDS object
    teds = TEDS(n_jobs=1)

    with open(predFile, 'rb') as f:
        predDict = pickle.load(f)

    with open(gtJsonFile, 'r') as f:
        gtValDict = json.load(f)

    assert len(predDict) == len(gtValDict) == 9115

    # # cut 10 to debug
    # file_names = [p for p in predDict.keys()][:10]
    # cut_predDict = dict()
    # for file_name in file_names:
    #     cut_predDict.setdefault(file_name, predDict[file_name])
    # predDict = cut_predDict

    scores = []
    caches = dict()
    for idx, (file_name, context) in enumerate(predDict.items()):
        # loading
        # file_name = os.path.basename(file_path)
        gt_context = gtValDict[file_name]
        # print(file_name)
        score = pool.apply_async(func=singleEvaluation, args=(teds, file_name, context, gt_context,))
        scores.append(score)
        tmp = {'score':score, 'gt':gt_context, 'pred':context}
        caches.setdefault(file_name, tmp)

    pool.close()
    pool.join() # 进程池中进程执行完毕后再关闭，如果注释，那么程序直接关闭。
    pool.terminate()

    # get score from scores
    cal_scores = []
    for score in scores:
        cal_scores.append(score.get())

    print('AVG TEDS score: {}'.format(sum(cal_scores)/len(cal_scores)))
    print('TEDS cost time: {}s'.format(time.time()-start_time))

    print("Save cache for analysis.")
    save_folder = '/data/ted_caches'
    for file_name in caches.keys():
        info = caches[file_name]
        if info['score']._value < 1.0:
            f = open(os.path.join(save_folder, file_name.replace('.png', '.txt')), 'w')
            f.write(file_name+'\n'+'\n')
            f.write('Score:'+'\n')
            f.write(str(info['score']._value)+'\n'+'\n')
            f.write('Pred:'+'\n')
            f.write(info['pred']+'\n'+'\n')
            f.write('Gt:' + '\n')
            f.write(info['gt']+'\n'+'\n')





