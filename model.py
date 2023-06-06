from utils import *




#### create LCDdata class: 
# query1, query2, query_seq_1, query_seq_2, seq1_timestamp, seq2_timestamp, gt
class LCDdata():
    def __init__(self, seq1_imgfolder, seq2_imgfolder, 
                 seq1, seq2, q1_pos, q2_pos, 
                seq1_relat_dist, seq2_relat_dist,
                 seq1_relat_time, seq2_relat_time, gt:float) -> None:
        self.seq1_imgfolder = seq1_imgfolder
        self.seq2_imgfolder = seq2_imgfolder
        
        self.seq1 = seq1
        self.seq2 = seq2    
            
        self.q1_pos = q1_pos
        self.q2_pos = q2_pos
        
        self.seq1_relat_dist = seq1_relat_dist
        self.seq2_relat_dist = seq2_relat_dist
        
        self.seq1_relat_time = seq1_relat_time
        self.seq2_relat_time = seq2_relat_time
        
        self.seq1_imgs, self.seq2_imgs = None, None
        self.edges1, self.edges2 = None, None
        
        ### init process
        self.__construct_edges()
        self.gt = torch.tensor([gt])    # float        
        self.seq1_relat_dist = torch.tensor(self.seq1_relat_dist).reshape((-1,1))
        self.seq1_relat_time = torch.tensor(self.seq1_relat_time).reshape((-1,1))        
        self.seq2_relat_dist = torch.tensor(self.seq2_relat_dist).reshape((-1,1))
        self.seq2_relat_time = torch.tensor(self.seq2_relat_time).reshape((-1,1))    
        # recommended modification: 
        # self.seq1_relat_dist = torch.tensor(xxx).clone().detach() 
        # or torch.tensor(xxx).clone().detach().requires_grad_(True)            
        
    def to_torch_seq(self,):
        read_imgs_1 = []
        read_imgs_2 = []
        for imgstr1 in self.seq1:
            whole_imgstr1 = os.path.join(self.seq1_imgfolder, imgstr1)
            read_imgs_1.append(read_image(whole_imgstr1).unsqueeze(0)) 
        for imgstr2 in self.seq2:
            whole_imgstr2 = os.path.join(self.seq2_imgfolder, imgstr2)
            read_imgs_2.append(read_image(whole_imgstr2).unsqueeze(0))                  
        self.seq1_imgs = torch.cat(read_imgs_1, dim=0)
        self.seq2_imgs = torch.cat(read_imgs_2, dim=0)

    def to_device(self, device):
        self.seq1_imgs = self.seq1_imgs.to(device)
        self.seq2_imgs = self.seq2_imgs.to(device)
        self.gt = self.gt.to(device)        
        self.seq1_relat_dist = self.seq1_relat_dist.to(device)
        self.seq1_relat_time = self.seq1_relat_time.to(device)       
        self.seq2_relat_dist = self.seq2_relat_dist.to(device)
        self.seq2_relat_time = self.seq2_relat_time.to(device)         

    def release_images(self,):
        self.seq1_imgs = None
        self.seq2_imgs = None

        
        
        
    def __construct_edges(self,):
        '''construct edges for graph using q_pos'''
        edges1 = [[],[]]
        for i in range(len(self.seq1)-1):
            edges1[0].append(i)
            edges1[1].append(i+1)
            edges1[0].append(i+1)
            edges1[1].append(i)
        for i in range(len(self.seq1)):
            if( (i != self.q1_pos-1) and (i != self.q1_pos) and (i != self.q1_pos+1) ):
                edges1[0].append(i)
                edges1[1].append(self.q1_pos)
                edges1[0].append(self.q1_pos)
                edges1[1].append(i)
        edges1 = torch.tensor(edges1, dtype=torch.long)
        self.edges1, _ = add_self_loops(edges1)
                
        edges2 = [[],[]]
        for i in range(len(self.seq2)-1):
            edges2[0].append(i)
            edges2[1].append(i+1)
            edges2[0].append(i+1)
            edges2[1].append(i)
        for i in range(len(self.seq2)):
            if( (i != self.q2_pos-1) and (i != self.q2_pos) and (i != self.q2_pos+1) ):
                edges2[0].append(i)
                edges2[1].append(self.q2_pos)
                edges2[0].append(self.q2_pos)
                edges2[1].append(i)
        edges2 = torch.tensor(edges2, dtype=torch.long)
        self.edges2, _ = add_self_loops(edges2)

class LCDModel(torch.nn.Module):
    
    def __init__(self, backbone, backbone_preprocess,           
                 gnn, gnn_out_dim, context_vec_dim,
                 device,
                 dropout_rat = 0.2,
                 dist_enc_dim = 32,
                 time_enc_dim = 32
                 ) -> None:
        super().__init__()
        
        '''backbone and gnn should be initialized out of this scope'''
        
        self.device = device      
        
        self.activation = torch.nn.LeakyReLU().to(device)
        self.regularization = torch.nn.Dropout(dropout_rat).to(device)
        self.sigmoid = torch.nn.Sigmoid().to(device)        

        self.backbone = backbone.to(device)
        self.backbone_preprocess = backbone_preprocess
        self.dist_encoder = torch.nn.Linear(1,dist_enc_dim).to(device)
        self.time_encoder = torch.nn.Linear(1,time_enc_dim).to(device)
        self.gnn = gnn.to(device)
        self.context_vec_head = torch.nn.Linear(gnn_out_dim, context_vec_dim).to(device)
        
        # self.score_head = torch.nn.Linear(context_vec_dim*2, 1).to(device)
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(context_vec_dim*2, context_vec_dim),
            self.activation,
            self.regularization,
            # torch.nn.Linear(context_vec_dim, context_vec_dim),
            # self.activation,
            # self.regularization, 
            torch.nn.Linear(context_vec_dim, 1),                                 
        )   
        self.score_head.to(device)

    
    @staticmethod
    def count_gnn_in_dim(backbone_out_dim, dist_enc_dim, time_enc_dim):
        return backbone_out_dim + dist_enc_dim + time_enc_dim
    
    # @staticmethod
    # def count_score_head_dim(gnn_out_dim):
    #     return gnn_out_dim * 2
    
    def forward(self, data:LCDdata):
        '''dimentions should be consistant'''  
        ### basic encodings
        batch_img, batch_relat_dist, batch_relat_time = self._preprocess(data)
        
        img_enc = self.backbone(batch_img).squeeze(0)   # shape: (n1+n2, img hidden)
        relat_dist_enc = self.dist_encoder(batch_relat_dist)  # shape: (n1+n2, dist hiddim)
        relat_time_enc = self.time_encoder(batch_relat_time)  # shape: (n1+n2, time hiddim) 
                 
        ### graph learning  
        seq1_enc, seq2_enc = self._midprocess(data, img_enc, relat_dist_enc, relat_time_enc) # shape: (n1 or n2, img+dist+time)
        graph1 = geo_Data(seq1_enc, data.edges1).to(self.device)
        graph2 = geo_Data(seq2_enc, data.edges2).to(self.device)
        gnn_enc_1 = self.gnn(graph1) 
        gnn_enc_2 = self.gnn(graph2)
        
        target_q1_enc = gnn_enc_1[data.q1_pos]
        target_q2_enc = gnn_enc_2[data.q2_pos]
        
        ### final matching score
        context_vec_1 = self.regularization(self.activation(self.context_vec_head(target_q1_enc)))
        context_vec_2 = self.regularization(self.activation(self.context_vec_head(target_q2_enc)))
        # shape: (context_dim,)
        
        final_enc = torch.cat((context_vec_1, context_vec_2), dim=0) # shape: (2*context_vec_dim)
        score = self.sigmoid(self.score_head(final_enc))    # shape: 1
        
        return score
        
    def _preprocess(self, data:LCDdata):
        '''form two sequence into batches'''
        # data.to_torch_seq()   # this has been moved outside
        
        batch_img = torch.cat((data.seq1_imgs, data.seq2_imgs), dim=0)
        batch_img = self.backbone_preprocess(batch_img).to(self.device)
        batch_relat_dist = torch.cat((data.seq1_relat_dist, data.seq2_relat_dist), dim=0).to(self.device)
        batch_relat_time = torch.cat((data.seq1_relat_time, data.seq2_relat_time), dim=0).to(self.device)
               
        return batch_img, batch_relat_dist, batch_relat_time
    
    def _midprocess(self, data:LCDdata, 
                     img_enc, relat_dist_enc, relat_time_enc):
        '''
        split the encoders and feed into gnn separately;
        also, add positional/temporal info
        '''
        
        fused_enc = torch.cat((img_enc,relat_dist_enc,relat_time_enc), dim=1).to(self.device)
        # shape of fused_enc should be (n1+n2, img+dist+time hidden)
        
        seq1_enc = fused_enc[:len(data.seq1)]   # n1,
        seq2_enc = fused_enc[len(data.seq1):]   # n2,
        
        return seq1_enc, seq2_enc
    
    # def _postprocess(self, data:LCDdata, target_enc_1, target_enc_2):
    #     '''fuse two feature and feed into scoring head'''
        


class GCN(torch.nn.Module):
    def __init__(self, layers, in_dim, hid_dim, out_dim, dropout_rat=0.2):
        super().__init__()
        
        assert layers>=2   
        conv_layers = [GCNConv(in_dim, hid_dim)]
        for i in range(layers-2):
            conv_layers.append(GCNConv(hid_dim,hid_dim))
        conv_layers.append(GCNConv(hid_dim, out_dim))
        
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.activation = torch.nn.LeakyReLU()
        self.regularization = torch.nn.Dropout(dropout_rat)          

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv_layer in self.conv_layers:
            x = self.activation(conv_layer(x, edge_index))
            x = self.regularization(x)

        return x

class GIN(torch.nn.Module):
    def __init__(self, layers, in_dim, hid_dim, out_dim, dropout_rat=0.2) -> None:
        super().__init__()
        
        assert layers>=2   
        
        gin_h_funcs = [torch.nn.Linear(in_dim, hid_dim)]            
        conv_layers = [GINConv(gin_h_funcs[0], train_eps=True)]
        for i in range(layers-2):
            gin_h = torch.nn.Linear(hid_dim, hid_dim)
            gin_h_funcs.append(gin_h)
            conv_layers.append(GINConv(gin_h, train_eps=True))
        gin_h = torch.nn.Linear(hid_dim, out_dim)
        gin_h_funcs.append(gin_h)
        conv_layers.append(GINConv(gin_h, train_eps=True))    
              
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.activation = torch.nn.LeakyReLU()
        self.regularization = torch.nn.Dropout(dropout_rat)        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv_layer in self.conv_layers:
            x = self.activation(conv_layer(x, edge_index))
            x = self.regularization(x)

        return x
    
class GATv2(torch.nn.Module):
    def __init__(self, layers, in_dim, hid_dim, out_dim, dropout_rat=0.2) -> None:
        super().__init__()
        
        assert layers>=2   
            
        conv_layers = [GATv2Conv(in_dim, hid_dim)]
        for i in range(layers-2):
            conv_layers.append(GATv2Conv(hid_dim,hid_dim))
        conv_layers.append(GATv2Conv(hid_dim, out_dim))
        
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.activation = torch.nn.LeakyReLU()
        self.regularization = torch.nn.Dropout(dropout_rat)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv_layer in self.conv_layers:
            x = self.activation(conv_layer(x, edge_index))
            x = self.regularization(x)

        return x