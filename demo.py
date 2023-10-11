import os
import sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, 'module/text_classification'))
from module.text_classification.inference import TextClassifier


config = {
    'checkpoint_lst': [os.path.join(root, 'model/rhetoric_model/Shewen_PretrainedBert_1e-05_32_None.pt')],
    'use_bert': True,
    'embd_path': os.path.join(root, 'model/word_embedding/tencent_small'),
    'model_config_lst': [{
        'is_state': False,
        'model_name': 'bert',
        'pretrained_model_path': os.path.join(root, 'model/bert_chinese_wwm_ext_pytorch')
    }]
}

def predict_sentences(sent_list, config):
    pretrained_model_path = config['model_config_lst'][0]['pretrained_model_path']
    model = TextClassifier(config['embd_path'], config['checkpoint_lst'], config['model_config_lst'],
                           pretrained_model_path)
    max_seq_len =config['max_seq_len'] if 'max_seq_len' in config else 80
    need_mask = config['need_mask'] if 'need_mask' in config else False
    pred_list, proba_list = model.predict_all_mask(sent_list, max_seq_len=max_seq_len, max_batch_size=20,
                                                   need_mask=need_mask)
    pos_sent_list = [sent_list[i] for i in range(len(pred_list)) if pred_list[i] == 1]
    
    print(pred_list, proba_list)
    print("设问修辞手法的数目", len(pos_sent_list))



if __name__ == "__main__":
    sent_list = [
        '你知道这个世界上最美的事物是什么吗？是人心',
        '为什么有些人总是不断地追求金钱和权力，却忽略了生命中更重要的东西？我想，这是因为他们没有意识到，人生的意义在于奉献，而不是索取。',
        '如果你有机会改变过去，你会选择改变什么？我选择改变自己，让自己变得更好。',
    ]
    # 调用函数：
    predict_sentences(sent_list,config)


