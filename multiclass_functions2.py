import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np # confusion matrix 사용시
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def Train(model, train_DL, val_DL, criterion, optimizer,
          EPOCH, BATCH_SIZE, 
          save_model_path, save_history_path, **kwargs):

    if "LR_STEP" in kwargs:
        scheduler = StepLR(optimizer, step_size=kwargs["LR_STEP"], gamma=kwargs["LR_GAMMA"])
    else: 
        scheduler = None
    
    loss_history = {"train": [], "val": []}
    acc_history = {"train": [], "val": []}
    best_loss = 9999
    for ep in range(EPOCH):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {ep+1}, current_LR = {current_lr}")
        
        model.train() # train mode로 전환
        train_loss, train_acc, _ = loss_epoch(model, train_DL, criterion, optimizer = optimizer)
        loss_history["train"] += [train_loss]
        acc_history["train"] += [train_acc]
        model.eval() # test mode로 전환
        with torch.no_grad():
            val_loss, val_acc, _ = loss_epoch(model, val_DL, criterion) # 이 때는 backpropagation 안하려고 optimizer안넣어줌!
            loss_history["val"] += [val_loss]
            acc_history["val"] += [val_acc]
            if val_loss < best_loss:
                best_loss = val_loss
                # optimizer도 같이 save하면 여기서부터 재학습 시작 가능
                torch.save({"model": model,
                            "ep": ep,
                            "optimizer": optimizer, # adam 을 사용하기 때문에 optimizaer도 저장해야한다! 
                            "scheduler": scheduler}, save_model_path) # learning rate도 decaying 하고있기 때문에 저장 해놓고 사용 
        if "LR_STEP" in kwargs:
            scheduler.step()
        # print loss
        print(f"train loss: {train_loss:.5f}, "
              f"val loss: {val_loss:.5f} \n"
              f"train acc: {train_acc:.1f} %, "
              f"val acc: {val_acc:.1f} %, time: {time.time()-epoch_start:.0f} s")
        print("-"*20)

    torch.save({"loss_history": loss_history,
                "acc_history": acc_history,
                "EPOCH": EPOCH,
                "BATCH_SIZE": BATCH_SIZE}, save_history_path)

def Test(model,test_DL, criterion):
    model.eval() # test mode로 전환
    with torch.no_grad():
        test_loss, test_acc, rcorrect = loss_epoch(model, test_DL, criterion)
    print()
    print(f"Test loss: {test_loss:.3f}")
    print(f"Test accuracy: {rcorrect}/{len(test_DL.dataset)} ({round(test_acc,1)} %)")
    return round(test_acc,1)
    
def loss_epoch(model, DL, criterion, optimizer = None):
    N = len(DL.dataset) # the number of data
    rloss = 0; rcorrect = 0
    for x_batch, y_batch in tqdm(DL, leave=False): #tqdm(DL, position=10, leave=False): # position은 줄바꿈 개수
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        # inference
        y_hat = model(x_batch)
        # loss
        loss = criterion(y_hat, y_batch)
        # update
        if optimizer is not None:
            optimizer.zero_grad() # gradient 누적을 막기 위한 초기화
            loss.backward() # backpropagation
            optimizer.step() # weight update
        # loss accumulation
        loss_b = loss.item() * x_batch.shape[0] # batch loss # BATCH_SIZE 로 하면 마지막 16개도 32개로 계산해버림
        rloss += loss_b # running loss
        # corrects accumulation
        pred = y_hat.argmax(dim=1)
        corrects_b = torch.sum(pred == y_batch).item()
        rcorrect += corrects_b
    loss_e = rloss/N # epoch loss
    accuracy_e = rcorrect/N * 100

    return loss_e, accuracy_e, rcorrect

def Test_plot(model, test_DL):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_DL))
        x_batch = x_batch.to(DEVICE)
        y_hat = model(x_batch)
        pred = y_hat.argmax(dim=1)

    x_batch = x_batch.to("cpu")

    plt.figure(figsize=(8,4))
    for idx in range(6):
        plt.subplot(2,3, idx+1, xticks=[], yticks=[])
        plt.imshow(x_batch[idx].permute(1,2,0), cmap="gray")
        pred_class = test_DL.dataset.classes[pred[idx]]
        true_class = test_DL.dataset.classes[y_batch[idx]]
        plt.title(f"{pred_class} ({true_class})", color = "g" if pred_class==true_class else "r") 

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num






















