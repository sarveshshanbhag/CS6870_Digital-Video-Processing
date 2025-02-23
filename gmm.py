import sys
import os
import cv2
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class GMM():
    def __init__(self,K,T,decay_rate,height,width):
        self.K=K
        self.T=T
        self.decay_rate=decay_rate
        self.height=height
        self.width=width
        self.mu=np.random.randint(0,256,(self.height,self.width,self.K,3))
        self.var=np.random.randint(10,40,(self.height,self.width,self.K))
        self.w=np.full((self.height,self.width,self.K),1/self.K)

    def classifier(self):
        bg_last=np.zeros((self.height,self.width),dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                bg_last[i,j]=-1
                ratios=[]
                for k in range(self.K):
                    ratios.append(self.w[i,j,k]/np.sqrt(self.var[i,j,k]+(1e-6)))
                indices=np.array(np.argsort(ratios)[::-1])
                self.mu[i,j]=self.mu[i,j][indices]
                self.var[i,j]=self.var[i,j][indices]
                self.w[i,j]=self.w[i,j][indices]
                sum_w=0
                for l in range(self.K-1):
                    sum_w=sum_w+self.w[i,j,l]
                    if sum_w>=self.T:
                        bg_last[i,j]=l
                        break
                if bg_last[i,j]==-1:
                    bg_last[i,j]=self.K-2
        return bg_last
    
    def update_parameters(self,current_frame,bg_last):
        fg_pixel=np.zeros((self.height,self.width))
        for i in range(self.height):
            for j in range(self.width):
                X=current_frame[i,j]
                match=-1
                for k in range(self.K):
                    inv_covar = np.diag([1.0 / (self.var[i,j,k]+(1e-6))] * 3)
                    distance=np.dot((X-self.mu[i,j,k]).T,np.dot(inv_covar,(X-self.mu[i,j,k])))
                    if distance<(2.5*2.5*self.var[i,j,k]):
                        match=k
                        break
                if match>-1:
                    self.w[i,j]=self.decay_rate*self.w[i,j]
                    self.w[i,j,match]=self.w[i,j,match]+(1-self.decay_rate)
                    rho=(1-self.decay_rate) * multivariate_normal.pdf(X,self.mu[i,j,match],np.diag([self.var[i,j,k]] * 3))
                    self.mu[i,j,match]=((rho)*X)+((1-rho)*self.mu[i,j,match])
                    self.var[i,j,match]=(rho)*(np.dot((X-self.mu[i,j,match]).T, (X-self.mu[i,j,match])))+((1-rho)*self.var[i,j,match])
                    if match>bg_last[i,j]:
                        fg_pixel[i,j]=255
                else:
                    self.mu[i,j,-1]=X
                    fg_pixel[i,j]=255
        return fg_pixel
    
    def output(self,frames_folder,output_folder):
        frame_count=0
        frame_files=sorted([f for f in os.listdir(frames_folder)])
        for frame_file in frame_files:
            frame_count+=1
            frame_path=os.path.join(frames_folder,frame_file)
            frame=cv2.imread(frame_path)
            print("Processing Frame: ", frame_file)
            if frame_count%20==0:
                print("Processing Frame (if): ", frame_file)
                bg_last=self.classifier()
                fg_pixels=self.update_parameters(frame,bg_last)
                cv2.imshow('background subtraction',fg_pixels)
                output_path = os.path.join(output_folder, f"res_{frame_count}.png")
                cv2.imwrite(output_path, fg_pixels)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def calculate_metrics(ground_truth, prediction):
    gt = ground_truth.flatten()
    pred = prediction.flatten()
    TP = np.sum((gt == 255) & (pred == 255))
    FP = np.sum((gt == 0) & (pred == 255))
    FN = np.sum((gt == 255) & (pred == 0))
    TN = np.sum((gt == 0) & (pred == 0))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    total_pixels = len(gt)
    accuracy = (TP + TN) / total_pixels if total_pixels > 0 else 0.0
    return precision, recall, f1_score, iou, accuracy


def compare_all(ground_truth_folder, prediction_folder):
    pred_files = sorted([f for f in os.listdir(prediction_folder) if f.endswith('.png')],
                        key=lambda x: int(x.split('_')[1].split('.')[0])) 
    metrics = []
    frame_numbers=[]
    for pred_file in pred_files:
        frame_number = int(pred_file.split('_')[1].split('.')[0])
        gt_file = f"gt{frame_number:06d}.png"
        gt_path = os.path.join(ground_truth_folder, gt_file)
        pred_path = os.path.join(prediction_folder, pred_file)
        if not os.path.exists(gt_path):
            print(f"Ground truth file {gt_file} not found. Skipping...")
            continue
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        assert gt_image.shape == pred_image.shape, f"Dimension mismatch: {gt_file} and {pred_file}"
        precision, recall, f1_score, iou, accuracy = calculate_metrics(gt_image, pred_image)
        metrics.append((pred_file, precision, recall, f1_score, iou, accuracy))
        frame_numbers.append(frame_number)
        print(f"Frame: {pred_file} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}, IoU: {iou:.4f}, Accuracy: {accuracy:.4f}")
    return frame_numbers, metrics

def plot_metrics(frame_numbers, metrics,save_path):
    plot_file = os.path.join(save_path, "metrics_plot.png")
    results_file = os.path.join(save_path, "average_metrics.txt")
    precisions = [m[1] for m in metrics]
    recalls = [m[2] for m in metrics]
    f1_scores = [m[3] for m in metrics]
    ious = [m[4] for m in metrics]
    accuracies = [m[5] for m in metrics]
    
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 2, 1)
    plt.plot(frame_numbers, precisions, label='Precision', color='blue')
    plt.xlabel('Frame Number')
    plt.ylabel('Precision')
    plt.title('Precision Over Frames')
    plt.grid()

    plt.subplot(3, 2, 2)
    plt.plot(frame_numbers, recalls, label='Recall', color='green')
    plt.xlabel('Frame Number')
    plt.ylabel('Recall')
    plt.title('Recall Over Frames')
    plt.grid()

    plt.subplot(3, 2, 3)
    plt.plot(frame_numbers, f1_scores, label='F1-Score', color='purple')
    plt.xlabel('Frame Number')
    plt.ylabel('F1-Score')
    plt.title('F1-Score Over Frames')
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.plot(frame_numbers, ious, label='IoU', color='orange')
    plt.xlabel('Frame Number')
    plt.ylabel('IoU')
    plt.title('IoU Over Frames')
    plt.grid()

    plt.subplot(3, 2, 5)
    plt.plot(frame_numbers, accuracies, label='Accuracy', color='red')
    plt.xlabel('Frame Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Frames')
    plt.grid()

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300) 
    plt.close() 

    avg_precision = np.mean([m[1] for m in metrics])
    avg_recall = np.mean([m[2] for m in metrics])
    avg_f1_score = np.mean([m[3] for m in metrics])
    avg_iou = np.mean([m[4] for m in metrics])
    avg_accuracy = np.mean([m[5] for m in metrics])
    with open(results_file, "w") as f:
        f.write("Overall Metrics:\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
        f.write(f"Average F1-Score: {avg_f1_score:.4f}\n")
        f.write(f"Average IoU: {avg_iou:.4f}\n")
        f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")

    print(f"Plot saved at: {plot_file}")
    print(f"Results saved at: {results_file}")

    print("\nOverall Metrics:")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1_score:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")

if __name__=='__main__':
    # gmm_model=GMM(K=5,T=0.6,decay_rate=0.7,height=480,width=720)
    frames_folder=r"C:\Users\91983\Downloads\ChangeDetection\ChangeDetection\pedestrians\input"
    prediction_folder=r"C:\Users\91983\Downloads\ChangeDetection\ChangeDetection\pedestrians\output"
    ground_truth_folder=r"C:\Users\91983\Downloads\ChangeDetection\ChangeDetection\pedestrians\groundtruth"
    # gmm_model.output(frames_folder,prediction_folder)
    frame_numbers, metrics = compare_all(ground_truth_folder, prediction_folder)
    plot_metrics(frame_numbers,metrics,prediction_folder)