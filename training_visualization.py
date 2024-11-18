import visdom
import re
import os
from time import sleep
import pandas as pd

npu_num = 8
log_file_name = r"/home/HwHiAiUser/read/line.log"

class iteration_log():
    def __init__(self, iteration):
        self.iteration = iteration
        self.loss = float('nan')
        self.elapsed_time = float('nan') #单位：ms
        self.learning_rate = float('nan')
        self.grad_norm = float('nan')
        self.consumed_samples = 0
        self.TGS = float('nan')
        
def table_content(content, content_type):
    if content_type == 'th':
        return '<th align="center">&nbsp' + str(content) + '&nbsp</th>'
    else:
        return '<' + content_type + ' align="center">' + str(content) + '</' + content_type +'>'        

def visualiza(log_file_name, npu_num, show_loss_line = True, show_table = True, show_elapsed_time = False):
    train_iters = 0
    global_batch_size = 0
    seq_length = 0
    viz_init = False
    
    with open(log_file_name, "r") as f:
        while True:
            if os.path.getsize(log_file_name) > f.tell():
                line = (f.readline()).strip()
                
                
                if line.find("iteration") == 0:
                    if not viz_init:
                        viz = visdom.Visdom(env='main')
                        if show_loss_line:
                            viz.line(X=[0.0], Y=[0.0], 
win="line_loss", name="loss", update='append', opts={'showlegend': True, 'title': "Loss曲线", 'xlabel': "iteration", 'ylabel': "loss",})

                        if show_elapsed_time:
                            viz.line(X=[0.0], Y=[0.0], 
win="line_elapsed_time", name="elapsed time", update='append', opts={'showlegend': True, 'title': "elapsed time", 'xlabel': "iteration", 'ylabel': "elapsed time per iteration (ms)",})

                        if show_table:
                            table_head = '<table border="1"><tr>' + table_content('iteration', 'th') + table_content('loss', 'th') + table_content('elapsed time(ms)', 'th') + table_content('TGS', 'th') + '</tr>'
                            table_rows = ''
                            table_tail = '</table>'  
                              
                        viz_init = True
                    
                    line_text = line.split('|')
                    iteration = int(re.search(r'\d+$', line_text[0][0:line_text[0].find('/')]).group(0))
                    iteration_mapping[iteration].loss = float(re.search(r'\d+\.\d+[E][+,-]\d+$', line_text[5].strip()).group(0))
                    iteration_mapping[iteration].elapsed_time = float(re.search(r'\d+\.\d+$', line_text[2].strip()).group(0))
                    iteration_mapping[iteration].learning_rate = float(re.search(r'\d+\.\d+[E][+,-]\d+$', line_text[3].strip()).group(0))
                    iteration_mapping[iteration].grad_norm = float(re.search(r'\d+\.\d+$', line_text[7].strip()).group(0))
                    iteration_mapping[iteration].consumed_samples = int(re.search(r'\d+$', line_text[1].strip()).group(0))
                    iteration_mapping[iteration].TGS = global_batch_size * seq_length / npu_num / iteration_mapping[iteration].elapsed_time * 1000
                    
                    if show_loss_line:
                        viz.line(X=[iteration], Y=[iteration_mapping[iteration].loss], name="loss", win='line_loss', update='append')
                    if show_elapsed_time:
                        viz.line(X=[iteration], Y=[iteration_mapping[iteration].elapsed_time], name="elapsed time", win='line_elapsed_time', update='append')
                    if show_table:
                        table_row = '<tr>' + table_content(iteration, 'td') + table_content(iteration_mapping[iteration].loss, 'td') + table_content(iteration_mapping[iteration].elapsed_time, 'td') + table_content(round(iteration_mapping[iteration].TGS, 2), 'td') + '</tr>'
                        table_rows = table_row + table_rows 
                        viz.text(table_head+table_rows+table_tail, win="table")
                    
                    if iteration == train_iters:
                        break
                        
                elif line.find("train_iters") == 0:
                    train_iters = int(re.search(r'\d*$', line).group(0))
                    iteration_mapping = { i: iteration_log(i) for i in range(1,train_iters + 1)}
                    
                elif line.find("global_batch_size") == 0:
                    global_batch_size = int(re.search(r'\d*$', line).group(0))
                
                elif line.find("seq_length") == 0:
                    seq_length = int(re.search(r'\d*$', line).group(0))
            
            else:
                sleep(0.5)

    return iteration_mapping


def table_to_csv(log_file_name):
    df = pd.DataFrame(columns=['iteration','loss','elapsed time(ms)','TGS'])
    for iteration in range(1, len(iteration_mapping) + 1):
        df.loc[len(df)] = [iteration, iteration_mapping[iteration].loss, iteration_mapping[iteration].elapsed_time, iteration_mapping[iteration].TGS]

    df['iteration'] = df['iteration'].astype(int)
    row_avg = ['avg', df['iteration'].mean(), df['loss'].mean(), df['elapsed time(ms)'].mean(), df['TGS'].mean()]
    row_var = ['var', df['iteration'].var(), df['loss'].var(), df['elapsed time(ms)'].var(), df['TGS'].var()]
    row_std = ['std', df['iteration'].std(), df['loss'].std(), df['elapsed time(ms)'].std(), df['TGS'].std()]
    row_max = ['max', df['iteration'].max(), df['loss'].max(), df['elapsed time(ms)'].max(), df['TGS'].max()]
    row_min = ['min', df['iteration'].min(), df['loss'].min(), df['elapsed time(ms)'].min(), df['TGS'].min()]
    
    df.loc[len(df)] = row_avg
    df.loc[len(df)] = row_var
    df.loc[len(df)] = row_std
    df.loc[len(df)] = row_max
    df.loc[len(df)] = row_min
    
    df.to_csv(log_file_name[0:log_file_name.find('.log')]+'_iterations.csv', index=False, na_rep='NaN')
    
    return df


if __name__ == "__main__":
    iteration_mapping = visualiza(log_file_name, npu_num)
    df = table_to_csv(log_file_name)
    
    
