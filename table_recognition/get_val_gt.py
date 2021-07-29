import os
import json
import json_lines
from tqdm import tqdm

def searchMerge(tokensList):
    pointer = 0
    mergedTokenList = []
    while tokensList[pointer]!='</tbody>':
        if tokensList[pointer]=='<td>':
            tmp=tokensList[pointer]+tokensList[pointer+1]
            mergedTokenList.append(tmp)
            pointer+=2
        elif tokensList[pointer]=='<td':  # <td colspan></td>
            if tokensList[pointer+2].startswith(' colspan') or tokensList[pointer+2].startswith(' rowspan'):
                # pattern <td rowspan="2" colspan="3">
                tmp = tokensList[pointer]+tokensList[pointer+1]+tokensList[pointer+2]+tokensList[pointer+3]+tokensList[pointer+4]
                pointer+=5
            else:
                # pattern <td colspan="3">
                tmp=tokensList[pointer]+tokensList[pointer+1]+tokensList[pointer+2]+tokensList[pointer+3]
                pointer+=4
            mergedTokenList.append(tmp)
        else:
            mergedTokenList.append(tokensList[pointer])
            pointer += 1
    mergedTokenList.append('</tbody>')
    return mergedTokenList


jsonFile = 'PubTabNet_2.0.0.jsonl'
smallVal300 = '/data_0/yejiaquan/data/TableRecognition/val'
thisValList = os.listdir(smallVal300)
gtDict = dict()
with open(jsonFile, 'rb') as f:
    for item in tqdm(json_lines.reader(f)):
        """
            item's keys : ['filename', 'split', 'imgid', 'html']
                    item['html']'s keys : ['cells', 'structure']
                    item['html']['cell'] : list of dict
                        eg. [
                            {"tokens": ["<b>", "V", "a", "r", "i", "a", "b", "l", "e", "</b>"], "bbox": [1, 4, 27, 13]},
                            {"tokens": ["<b>", "H", "a", "z", "a", "r", "d", " ", "r", "a", "t", "i", "o", "</b>"], "bbox": [219, 4, 260, 13]},
                        ]
                    item['html']['structure']'s ['tokens']
                        eg. "structure": {"tokens": ["<thead>", "<tr>", "<td>", "</td>", ... ,"</tbody>"}
        """
        if item['split'] != 'val':
            continue
        filename = item['filename']
        esbFlag = False
        beFlag = False
        rawToken = item['html']['structure']['tokens']
        mergeTokenList = searchMerge(rawToken)
        mergeToken = ''.join(mergeTokenList)

        # text
        cells = item['html']['cells']
        textList = []
        for cell in cells:
            if len(cell['tokens']) == 0:
                # empty bbox
                textList.append('')
            else:
                textList.append(''.join(cell['tokens']))

        try:
            assert len(textList) == mergeToken.count('<td')
        except:
            # import pdb;pdb.set_trace()
            raise ValueError


        textCount = 0
        gtTokenList = []
        for mt in mergeTokenList:
            if mt.startswith('<td'):
                mt = mt.replace('><', '>{}<'.format(textList[textCount]))
                textCount = textCount + 1
            gtTokenList.append(mt)
        gtToken = ''.join(gtTokenList)
        gtDict.setdefault(filename, gtToken)


gtFile = 'gtVal_0726.json'
with open(gtFile, 'w') as f:
    json.dump(gtDict, f)