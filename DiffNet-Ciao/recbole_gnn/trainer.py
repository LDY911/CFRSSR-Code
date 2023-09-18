from time import time
import math
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from recbole.trainer import Trainer
from recbole.utils import early_stopping, dict2str, set_color, get_gpu_usage


class CIRecTrainer(Trainer):

    def __init__(self, config, model):
        super(CIRecTrainer, self).__init__(config, model)
        self.a = '自己的Trainer'

    # 训练
    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=True, callback_fn=None):
        # 存储模型参数信息和训练信息的
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)
        # 为evaluator的评估收集数据
        self.eval_collector.data_collect(train_data)  # 不用关心
        if self.config['train_neg_sample_args'].get('dynamic', 'none') != 'none':
            train_data.get_model(self.model)  # 用不到
        valid_step = 0
        valid_ft_step = 0
        # 训练
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            # 一个epoch内所有batch训练得到的总损失函数  +++++这里通过层层调用，最终调用了对应类的计算损失的函数+++++
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            # 记录每一代的总损失函数值
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            # 这里就是把训练的各种值在控制台显示出来
            train_loss_output = self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            # 将训练的损失记录在日志信息里
            if verbose:  # 是否要记录在日志信息里
                self.logger.info(train_loss_output)
            # 将训练的损失值添加到tensorboard里
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            # 记录权重和偏差的指标
            self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step': epoch_idx}, head='train')

            # 在验证集上评估(这里每训练一个epoch就评估一下)
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                # 使用验证集数据检验模型   ++++++++这里通过层层调用，最后调用了对应类的predict方法进行预测++++++++
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                # 这里就是实现超过十步不提高模型精度就停止迭代的操作
                # best_valid_score：最佳精度值
                # cur_step：从当前代开始多少代没有提升精度了
                # stop_flag：是否停止
                # update_flag：是否更新
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(valid_score, self.best_valid_score, self.cur_step, max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                # 下面这些就是把验证结果显示在控制台上，注：以各种评估指标中的mrr@10为valid_score
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue') + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                # 记录验证的各种信息
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                # 将验证集上的各种指标加到tensorboard里
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                # 记录权重和偏差的指标
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')
                # 是否继续更新
                if update_flag:
                    if saved:  # 存储信息
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result
                # 在迭代结束时执行
                if callback_fn:
                    callback_fn(epoch_idx, valid_score)
                # 是否停止更新
                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        # 将迭代后的超参数加入到tensorboard中
        self._add_hparam_to_tensorboard(self.best_valid_score)
        # 返回最佳检验结果和最佳各项指标
        return self.best_valid_score, self.best_valid_result

    # 工具函数
    def _train_ft_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        # 设置损失函数
        loss_func = loss_func or self.model.calculate_ft_loss  # 调用对应模型自己的计算损失的函数
        # 定义总损失值
        total_loss = None
        # 展示训练过程
        iter_data = (tqdm(train_data, total=len(train_data), ncols=100, desc=set_color(f"Train {epoch_idx:>5}", 'pink'), ) if show_progress else train_data)
        # 一个batch一个batch的循环
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)  # 部署
            self.optimizer.zero_grad()  # 梯度清零
            losses = loss_func(interaction)  # 计算损失  ++++++++这里，直接调用了对应类的计算损失的函数，而计算损失的函数又直接掉调用模型的forward函数，因此实现了模型的运行++++++++
            if isinstance(losses, tuple):  # 如果losses的类型是tuple(说明损失值由不同的part构成)
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)  # 检查损失值是否为nan(无穷大)
            loss.backward()  # 反向传播
            if self.clip_grad_norm:  # 是否需要做梯度修剪
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()  # 优化
            if self.gpu_available and show_progress:  # 使用GPU
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss



    # 工具函数
    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        # 设置为训练模式
        self.model.train()
        # 设置损失函数
        loss_func = loss_func or self.model.calculate_loss  # 调用对应模型自己的计算损失的函数
        # 定义总损失值
        total_loss = None
        # 展示训练过程
        iter_data = (tqdm(train_data, total=len(train_data), ncols=100, desc=set_color(f"Train {epoch_idx:>5}", 'pink'), ) if show_progress else train_data)
        # 一个batch一个batch的循环
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)  # 部署
            self.optimizer.zero_grad()  # 梯度清零
            losses = loss_func(interaction)  # 计算损失  ++++++++这里，直接调用了对应类的计算损失的函数，而计算损失的函数又直接掉调用模型的forward函数，因此实现了模型的运行++++++++
            if isinstance(losses, tuple):  # 如果losses的类型是tuple(说明损失值由不同的part构成)
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)  # 检查损失值是否为nan(无穷大)
            loss.backward()  # 反向传播
            if self.clip_grad_norm:  # 是否需要做梯度修剪
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()  # 优化
            if self.gpu_available and show_progress:  # 使用GPU
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss


class NCLTrainer(Trainer):
    def __init__(self, config, model):
        super(NCLTrainer, self).__init__(config, model)

        self.num_m_step = config['m_step']
        assert self.num_m_step is not None

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.
        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.
        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        self.eval_collector.data_collect(train_data)

        for epoch_idx in range(self.start_epoch, self.epochs):

            # only differences from the original trainer
            if epoch_idx % self.num_m_step == 0:
                self.logger.info("Running E-step ! ")
                self.model.e_step()
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = set_color('Saving current', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch
        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.
        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                if epoch_idx < self.config['warm_up_step']:
                    losses = losses[:-1]
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss


class HMLETTrainer(Trainer):
    def __init__(self, config, model):
        super(HMLETTrainer, self).__init__(config, model)

        self.warm_up_epochs = config['warm_up_epochs']
        self.ori_temp = config['ori_temp']
        self.min_temp = config['min_temp']
        self.gum_temp_decay = config['gum_temp_decay']
        self.epoch_temp_decay = config['epoch_temp_decay']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if epoch_idx > self.warm_up_epochs:
            # Temp decay
            gum_temp = self.ori_temp * math.exp(-self.gum_temp_decay*(epoch_idx - self.warm_up_epochs))
            self.model.gum_temp = max(gum_temp, self.min_temp)
            self.logger.info(f'Current gumbel softmax temperature: {self.model.gum_temp}')

            for gating in self.model.gating_nets:
                self.model._gating_freeze(gating, True)
        return super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)


class SEPTTrainer(Trainer):
    def __init__(self, config, model):
        super(SEPTTrainer, self).__init__(config, model)
        self.warm_up_epochs = config['warm_up_epochs']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if epoch_idx < self.warm_up_epochs:
            loss_func = self.model.calculate_rec_loss
        else:
            self.model.subgraph_construction()
        return super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)