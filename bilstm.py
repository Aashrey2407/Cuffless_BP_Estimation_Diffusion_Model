import torch
import torch.nn as nn
class BP_Estimator(nn.Module):
    def __init__(self, input_dim=2, hidden_size=128, num_layers=2, output_dim=2, dropout=0.2):
        """
        BiLSTM-based blood pressure estimator.
        :param input_dim: number of input channels (e.g., 2 if concatenating ECG and PPG)
        :param hidden_size: hidden state size for the LSTM
        :param num_layers: number of LSTM layers
        :param output_dim: number of outputs (2 for SBP and DBP)
        :param dropout: dropout probability between LSTM layers
        """
        super(BP_Estimator,self).__init__()
        self.bilstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout,
        )
        #fully connected layer
        self.fc = nn.Linear(hidden_size*2,output_dim)
        self.loss_fn = nn.MSELoss()

    def forward(self,generated_ecg,original_ppg,bp_target = None):
        """
        :param generated_ecg: Tensor of generated ECG signals of shape (batch, 1, T)
        :param original_ppg: Tensor of original PPG signals of shape (batch, 1, T)
        :param bp_target: (optional) Tensor of ground truth BP values of shape (batch, 2)
        :return: If bp_target is provided, returns (bp_pred, loss); else returns bp_pred.
        """
        #input to bilstm is generated ecg and original ppg
        x = torch.cat([generated_ecg,original_ppg],dim = 1)
        #transposing from (2,T) to (T,2) so that time is the sequence domain
        x = x.transpose(1,2)
        #pass input into the bilstm
        output,(h_n,_) = self.bilstm(x)
        #get the final forward and backward states
        h_forward = h_n[-2]
        h_backward = h_n[-1]

        #concatenate forward and backward states
        h_cat = torch.cat((h_forward,h_backward),dim = 1)
        bp_pred = self.fc(h_cat)

        if bp_target is not None:
            L_bp = self.loss_fn(bp_pred,bp_target)
            return bp_pred,L_bp
        
        return bp_pred
