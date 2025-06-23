import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import networkx as nx
import os
import warnings
warnings.filterwarnings('ignore')

class EnhancedFinancialDataProcessor:
    def __init__(self, metadata_path: str, price_data_paths: list):
        """
        Enhanced processor that handles multiple companies' data
        
        Args:
            metadata_path: "ind_nifty50list.csv"
            price_data_paths: [
                "nifty_stocks/ADANIENT.csv", 
                "nifty_stocks/ADANIPORTS.csv",
                "nifty_stocks/APOLLOHOSP.csv",
                "nifty_stocks/ASIANPAINT.csv",
                "nifty_stocks/AXISBANK.csv",
                "nifty_stocks/BAJAJFINSV.csv",
                "nifty_stocks/BAJFINANCE.csv",
                "nifty_stocks/BEL.csv",
                "nifty_stocks/BHARTIARTL.csv",
                "nifty_stocks/CIPLA.csv",
                "nifty_stocks/COALINDIA.csv",
                "nifty_stocks/DRREDDY.csv",
                "nifty_stocks/EICHERMOT.csv",
                "nifty_stocks/GRASIM.csv",
                "nifty_stocks/HCLTECH.csv",
                "nifty_stocks/HDFCBANK.csv",
                "nifty_stocks/HDFCLIFE.csv",
                "nifty_stocks/HEROMOTOCO.csv",
                "nifty_stocks/HINDALCO.csv",
                "nifty_stocks/HINDUNILVR.csv",
                "nifty_stocks/ICICIBANK.csv",
                "nifty_stocks/INDUSINDBK.csv",
                "nifty_stocks/INFY.csv",
                "nifty_stocks/ITC.csv",
                "nifty_stocks/JIOFIN.csv",
                "nifty_stocks/JSWSTEEL.csv",
                "nifty_stocks/KOTAKBANK.csv",
                "nifty_stocks/LT.csv",
                "nifty_stocks/M&M.csv",
                "nifty_stocks/MARUTI.csv",
                "nifty_stocks/NESTLEIND.csv",
                "nifty_stocks/NTPC.csv",
                "nifty_stocks/ONGC.csv",
                "nifty_stocks/POWERGRID.csv",
                "nifty_stocks/RELIANCE.csv",
                "nifty_stocks/SBILIFE.csv",
                "nifty_stocks/SBIN.csv",
                "nifty_stocks/SHRIRAMFIN.csv",
                "nifty_stocks/SUNPHARMA.csv",
                "nifty_stocks/TATACONSUM.csv",
                "nifty_stocks/TATAMOTORS.csv",
                "nifty_stocks/TATASTEEL.csv",
                "nifty_stocks/TCS.csv",
                "nifty_stocks/TECHM.csv",
                "nifty_stocks/TITAN.csv",
                "nifty_stocks/TRENT.csv",
                "nifty_stocks/ULTRACEMCO.csv",
                "nifty_stocks/WIPRO.csv"
        ]
        """
        self.metadata = pd.read_csv(metadata_path)
        self.companies_data = {}
        self.all_price_data = pd.DataFrame()
        
        # Load all company price data
        for path in price_data_paths:
            if os.path.exists(path):
                company_data = self._load_company_data(path)
                if company_data is not None:
                    company_symbol = self._extract_symbol_from_filename(path)
                    self.companies_data[company_symbol] = company_data
                    
        # Combine all data for correlation analysis
        self._combine_all_data()
        self._prepare_company_mappings()
        
        print(f"Loaded data for {len(self.companies_data)} companies")
        print(f"Metadata shape: {self.metadata.shape}")
        
    def _load_company_data(self, path):
        """Load and clean individual company data"""
        try:
            data = pd.read_csv(path)
            data.columns = data.columns.str.strip().str.lower()
            data.rename(columns={
                'timestamp': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Shares Traded'
            }, inplace=True)

            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Shares Traded']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            data = data.fillna(method='ffill').fillna(method='bfill')
            return data.sort_values('Date').reset_index(drop=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    
    def _extract_symbol_from_filename(self, path):
        """Extract company symbol from filename"""
        filename = os.path.basename(path)
        return filename.split('-')[0]
    
    def _combine_all_data(self):
        """Combine all company data for correlation analysis"""
        combined_data = []
        for symbol, data in self.companies_data.items():
            temp_data = data.copy()
            temp_data['Symbol'] = symbol
            combined_data.append(temp_data)
        
        if combined_data:
            self.all_price_data = pd.concat(combined_data, ignore_index=True)
    
    def _prepare_company_mappings(self):
        """Create mappings between companies and their indices"""
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.companies_data.keys())}
        self.idx_to_symbol = {idx: symbol for symbol, idx in self.symbol_to_idx.items()}
        self.num_companies = len(self.companies_data)

    def prepare_features_for_all_companies(self, lookback: int = 20, forward: int = 5):
        """Prepare features for all companies"""
        all_features = []
        all_targets = []
        company_indices = []
        
        for symbol, data in self.companies_data.items():
            try:
                features, targets = self._prepare_single_company_features(data, lookback, forward)
                
                # Add company-specific features
                company_idx = self.symbol_to_idx[symbol]
                features['company_idx'] = company_idx
                
                # Add industry encoding
                company_industry = self._get_company_industry(symbol)
                features['industry_encoded'] = company_industry
                
                all_features.append(features)
                all_targets.append(targets)
                company_indices.extend([company_idx] * len(features))
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid company data processed")
        
        # Combine all features and targets
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_targets = pd.concat(all_targets, ignore_index=True)
        
        print(f"Total samples across all companies: {len(combined_features)}")
        print(f"Feature columns: {list(combined_features.columns)}")
        
        return combined_features, combined_targets, company_indices

    def _prepare_single_company_features(self, data, lookback, forward):
        """Prepare features for a single company"""
        data['returns'] = data['Close'].pct_change()
        
        # Technical indicators
        features_dict = {
            'returns_5d': data['returns'].rolling(5, min_periods=1).mean(),
            'returns_10d': data['returns'].rolling(10, min_periods=1).mean(),
            'returns_20d': data['returns'].rolling(20, min_periods=1).mean(),
            'volatility_5d': data['returns'].rolling(5, min_periods=1).std(),
            'volatility_10d': data['returns'].rolling(10, min_periods=1).std(),
            'volatility_20d': data['returns'].rolling(20, min_periods=1).std(),
            'volume_5d': data['Shares Traded'].rolling(5, min_periods=1).mean(),
            'volume_10d': data['Shares Traded'].rolling(10, min_periods=1).mean(),
            'rsi_14d': self._calculate_rsi(data['returns'], 14),
            'price_ma_5': data['Close'].rolling(5, min_periods=1).mean(),
            'price_ma_20': data['Close'].rolling(20, min_periods=1).mean(),
            'bb_upper': data['Close'].rolling(20).mean() + 2 * data['Close'].rolling(20).std(),
            'bb_lower': data['Close'].rolling(20).mean() - 2 * data['Close'].rolling(20).std(),
        }
        
        # Relative features
        features_dict['price_vs_ma5'] = data['Close'] / features_dict['price_ma_5'] - 1
        features_dict['price_vs_ma20'] = data['Close'] / features_dict['price_ma_20'] - 1
        features_dict['bb_position'] = (data['Close'] - features_dict['bb_lower']) / (features_dict['bb_upper'] - features_dict['bb_lower'])
        
        features = pd.DataFrame(features_dict)
        features['Date'] = data['Date']
        
        # Targets
        future_returns = data['Close'].pct_change(forward).shift(-forward)
        targets = pd.DataFrame({
            'Date': data['Date'],
            'price_target': future_returns,
            'direction_target': (future_returns > 0).astype(int)
        })
        
        # Clean data
        features = features.dropna()
        targets = targets.dropna()
        merged = pd.merge(features, targets, on='Date', how='inner')
        
        if len(merged) == 0:
            raise ValueError("No valid data after processing")
        
        feature_cols = [col for col in merged.columns if col not in ['Date', 'price_target', 'direction_target']]
        final_features = merged[feature_cols]
        final_targets = merged[['price_target', 'direction_target']]
        
        return final_features, final_targets
    
    def _get_company_industry(self, symbol):
        """Get industry encoding for a company"""
        if hasattr(self, 'industry_encoder'):
            company_row = self.metadata[self.metadata['Symbol'] == symbol]  
            if not company_row.empty:
                industry = company_row['Industry'].iloc[0]
                return self.industry_encoder.transform([industry])[0]
        return 0
    
    def _calculate_rsi(self, series, window):
        """Calculate RSI indicator"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window, min_periods=1).mean()
        avg_loss = loss.rolling(window, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_industry_encoder(self):
        """Create label encoder for industries"""
        self.industry_encoder = LabelEncoder()
        self.industry_encoder.fit(self.metadata['Industry'].dropna())
        print(f"Industries encoded: {list(self.industry_encoder.classes_)}")


class FinancialGraphBuilder:
    def __init__(self, metadata: pd.DataFrame, companies_data: dict, price_data: pd.DataFrame):
        self.metadata = metadata
        self.companies_data = companies_data
        self.price_data = price_data
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(companies_data.keys())}
        self.num_companies = len(companies_data)
        
    def build_comprehensive_graph(self, 
                                 industry_weight=0.4, 
                                 correlation_weight=0.3, 
                                 market_cap_weight=0.2,
                                 correlation_threshold=0.3):
        """
        Build a comprehensive financial graph with multiple relationship types
        """
        print("Building comprehensive financial graph...")
        
        # Initialize adjacency matrix
        adj_matrix = np.zeros((self.num_companies, self.num_companies))
        
        # 1. Industry relationships
        industry_adj = self._build_industry_relationships()
        adj_matrix += industry_weight * industry_adj
        
        # 2. Price correlation relationships
        correlation_adj = self._build_correlation_relationships(correlation_threshold)
        adj_matrix += correlation_weight * correlation_adj
        
        # 3. Market cap similarity relationships
        market_cap_adj = self._build_market_cap_relationships()
        adj_matrix += market_cap_weight * market_cap_adj
        
        # Convert to edge indices
        edge_indices = self._adjacency_to_edge_index(adj_matrix)
        
        # Create edge attributes (relationship strengths)
        edge_attrs = self._create_edge_attributes(adj_matrix, edge_indices)
        
        print(f"Graph created with {len(edge_indices[0])} edges")
        print(f"Average degree: {len(edge_indices[0]) / self.num_companies:.2f}")
        
        return {
            'edge_index': torch.tensor(edge_indices, dtype=torch.long),
            'edge_attr': torch.tensor(edge_attrs, dtype=torch.float),
            'num_nodes': self.num_companies
        }
    
    def _build_industry_relationships(self):
        """Build relationships based on industry similarity"""
        adj_matrix = np.zeros((self.num_companies, self.num_companies))
        
        for i, symbol_i in enumerate(self.companies_data.keys()):
            for j, symbol_j in enumerate(self.companies_data.keys()):
                if i != j:
                    industry_i = self._get_company_industry(symbol_i)
                    industry_j = self._get_company_industry(symbol_j)
                    
                    if industry_i == industry_j and industry_i is not None:
                        adj_matrix[i, j] = 1.0
        
        print(f"Industry relationships: {np.sum(adj_matrix > 0)} edges")
        return adj_matrix
    
    def _build_correlation_relationships(self, threshold=0.3):
        """Build relationships based on price correlation"""
        adj_matrix = np.zeros((self.num_companies, self.num_companies))
        
        # Calculate correlations between all pairs
        symbols = list(self.companies_data.keys())
        
        for i, symbol_i in enumerate(symbols):
            for j, symbol_j in enumerate(symbols):
                if i != j:
                    corr = self._calculate_price_correlation(symbol_i, symbol_j)
                    if abs(corr) > threshold:
                        adj_matrix[i, j] = abs(corr)
        
        print(f"Correlation relationships: {np.sum(adj_matrix > 0)} edges")
        return adj_matrix
    
    def _build_market_cap_relationships(self):
        """Build relationships based on market cap similarity"""
        adj_matrix = np.zeros((self.num_companies, self.num_companies))
        
        # Calculate market caps (approximate using turnover)
        market_caps = {}
        for symbol, data in self.companies_data.items():
            avg_turnover = data['Shares Traded'].mean()
            market_caps[symbol] = avg_turnover
        
        symbols = list(self.companies_data.keys())
        market_cap_values = [market_caps[symbol] for symbol in symbols]
        
        # Normalize market caps
        market_cap_values = np.array(market_cap_values)
        market_cap_values = (market_cap_values - np.mean(market_cap_values)) / np.std(market_cap_values)
        
        # Create relationships based on market cap similarity
        for i in range(len(symbols)):
            for j in range(len(symbols)):
                if i != j:
                    similarity = np.exp(-abs(market_cap_values[i] - market_cap_values[j]) / 2)
                    if similarity > 0.5:  # Threshold for market cap similarity
                        adj_matrix[i, j] = similarity
        
        print(f"Market cap relationships: {np.sum(adj_matrix > 0)} edges")
        return adj_matrix
    
    def _calculate_price_correlation(self, symbol1, symbol2):
        """Calculate price correlation between two companies"""
        try:
            data1 = self.companies_data[symbol1]['Close'].values
            data2 = self.companies_data[symbol2]['Close'].values
            
            # Align data lengths
            min_len = min(len(data1), len(data2))
            data1 = data1[-min_len:]
            data2 = data2[-min_len:]
            
            if len(data1) > 10:  # Minimum data points for correlation
                corr, _ = pearsonr(data1, data2)
                return corr if not np.isnan(corr) else 0
        except:
            pass
        return 0
    
    def _get_company_industry(self, symbol):
        """Get company industry from metadata"""
        company_row = self.metadata[self.metadata['Symbol'] == symbol]
        if not company_row.empty:
            return company_row['Industry'].iloc[0]
        return None
    
    def _adjacency_to_edge_index(self, adj_matrix):
        """Convert adjacency matrix to edge index format"""
        edges = np.nonzero(adj_matrix)
        return [edges[0].tolist(), edges[1].tolist()]
    
    def _create_edge_attributes(self, adj_matrix, edge_indices):
        """Create edge attributes from adjacency matrix"""
        edge_attrs = []
        for i, j in zip(edge_indices[0], edge_indices[1]):
            edge_attrs.append(adj_matrix[i, j])
        return edge_attrs
    
    def visualize_graph(self, graph_data, save_path='financial_graph.png'):
        """Visualize the financial graph"""
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        symbols = list(self.companies_data.keys())
        for i, symbol in enumerate(symbols):
            G.add_node(i, label=symbol)
        
        # Add edges
        edge_index = graph_data['edge_index'].numpy()
        edge_attr = graph_data['edge_attr'].numpy()
        
        for i, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
            G.add_edge(src, dst, weight=edge_attr[i])
        
        # Plot
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], alpha=0.6)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
        
        # Draw labels
        labels = {i: symbols[i] for i in range(len(symbols))}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title('Financial Network Graph')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class EnhancedFinancialGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_companies, num_heads=8, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.num_companies = num_companies
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-layer GAT
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=1)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=1)
                )
        
        # Company embedding
        self.company_embedding = nn.Embedding(num_companies, hidden_dim // 4)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Prediction heads
        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, company_indices, graph_data):
        batch_size = x.size(0)
        
        # Project input features
        x = F.relu(self.input_proj(x))
        
        # Get company embeddings
        company_indices_tensor = torch.tensor(company_indices, dtype=torch.long, device=x.device)
        company_emb = self.company_embedding(company_indices_tensor)
        
        # Create graph data for this batch
        edge_index = graph_data['edge_index'].to(x.device)
        edge_attr = graph_data['edge_attr'].to(x.device)
        
        # For each sample in the batch, we need to create a separate graph
        # This is a simplified approach - in practice, you might want to use batch processing
        batch_outputs = []
        
        for i in range(batch_size):
            # Get the company index for this sample
            company_idx = company_indices[i]
            
            # Create node features for the entire graph
            # Each node gets the same base features (this is simplified)
            node_features = x[i].unsqueeze(0).repeat(self.num_companies, 1)
            
            # Apply GAT layers
            gat_output = node_features
            for gat_layer in self.gat_layers:
                gat_output = F.relu(gat_layer(gat_output, edge_index, edge_attr.unsqueeze(-1)))
                gat_output = F.dropout(gat_output, training=self.training)
            
            # Get the output for the specific company
            company_output = gat_output[company_idx]
            
            # Fuse with company embedding
            fused_features = torch.cat([company_output, company_emb[i]], dim=0)
            fused_features = self.feature_fusion(fused_features)
            
            batch_outputs.append(fused_features)
        
        # Stack batch outputs
        x = torch.stack(batch_outputs, dim=0)
        
        # Predictions
        price_pred = self.price_head(x).squeeze(-1)
        direction_pred = self.direction_head(x).squeeze(-1)
        
        return price_pred, direction_pred


class EnhancedFinancialDataset(Dataset):
    def __init__(self, features, targets, company_indices):
        self.features = torch.FloatTensor(features.values)
        self.price_targets = torch.FloatTensor(targets['price_target'].values)
        self.direction_targets = torch.FloatTensor(targets['direction_target'].values)
        self.company_indices = company_indices
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'price_target': self.price_targets[idx],
            'direction_target': self.direction_targets[idx],
            'company_index': self.company_indices[idx]
        }


class EnhancedFinancialTrainer:
    def __init__(self, model, graph_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.graph_data = graph_data
        self.device = device
        self.best_val_loss = float('inf')
        
        # Move graph data to device
        self.graph_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in self.graph_data.items()}
        
        print(f"Using device: {device}")
        print(f"Graph has {self.graph_data['num_nodes']} nodes and {len(self.graph_data['edge_index'][0])} edges")
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
        
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss, train_batches = 0, 0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                price_target = batch['price_target'].to(self.device)
                direction_target = batch['direction_target'].to(self.device)
                company_indices = batch['company_index']
                
                optimizer.zero_grad()
                
                price_pred, direction_pred = self.model(features, company_indices, self.graph_data)
                
                price_loss = F.mse_loss(price_pred, price_target)
                direction_loss = F.binary_cross_entropy(direction_pred, direction_target)
                
                # Weighted loss
                loss = 0.7 * price_loss + 0.3 * direction_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            val_loss, val_mae, val_acc = self.validate(val_loader)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val MAE: {val_mae:.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'graph_data': self.graph_data,
                    'val_loss': val_loss,
                    'epoch': epoch
                }, 'best_enhanced_fingat_model.pth')
                print(f"🎯 New best model saved! Val Loss: {val_loss:.4f}")
        
        # Plot training curves
        self._plot_training_curves(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def validate(self, data_loader):
        self.model.eval()
        total_loss, val_batches = 0, 0
        price_preds, price_targets = [], []
        direction_preds, direction_targets = [], []
        
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                price_target = batch['price_target'].to(self.device)
                direction_target = batch['direction_target'].to(self.device)
                company_indices = batch['company_index']
                
                price_pred, direction_pred = self.model(features, company_indices, self.graph_data)
                
                price_loss = F.mse_loss(price_pred, price_target)
                direction_loss = F.binary_cross_entropy(direction_pred, direction_target)
                loss = 0.7 * price_loss + 0.3 * direction_loss
                
                total_loss += loss.item()
                val_batches += 1
                
                price_preds.extend(price_pred.cpu().numpy())
                price_targets.extend(price_target.cpu().numpy())
                direction_preds.extend(direction_pred.cpu().numpy())
                direction_targets.extend(direction_target.cpu().numpy())
        
        avg_val_loss = total_loss / val_batches
        mae = mean_absolute_error(price_targets, price_preds)
        acc = accuracy_score(direction_targets, np.round(direction_preds))
        
        return avg_val_loss, mae, acc
    
    def _plot_training_curves(self, train_losses, val_losses):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', alpha=0.8)
        plt.plot(val_losses, label='Validation Loss', alpha=0.8)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(np.log(train_losses), label='Log Train Loss', alpha=0.8)
        plt.plot(np.log(val_losses), label='Log Validation Loss', alpha=0.8)
        plt.title('Training and Validation Loss (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_fingat_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Main function to run the enhanced FinGAT model
    """
    try:
        print("🚀 Starting Enhanced FinGAT Training...")
        
        # 1. Prepare file paths (you'll need to adjust these)
        metadata_path = 'ind_nifty50list.csv'
        
        # You'll need to provide paths to individual company data files
        # For now, using a single file as example
        price_data_paths = [
            "nifty_stocks/ADANIENT.csv",
            "nifty_stocks/ADANIPORTS.csv",
                "nifty_stocks/APOLLOHOSP.csv",
                "nifty_stocks/ASIANPAINT.csv",
                "nifty_stocks/AXISBANK.csv",
                "nifty_stocks/BAJAJFINSV.csv",
                "nifty_stocks/BAJFINANCE.csv",
                "nifty_stocks/BEL.csv",
                "nifty_stocks/BHARTIARTL.csv",
                "nifty_stocks/CIPLA.csv",
                "nifty_stocks/COALINDIA.csv",
                "nifty_stocks/DRREDDY.csv",
                "nifty_stocks/EICHERMOT.csv",
                "nifty_stocks/GRASIM.csv",
                "nifty_stocks/HCLTECH.csv",
                "nifty_stocks/HDFCBANK.csv",
                "nifty_stocks/HDFCLIFE.csv",
                "nifty_stocks/HEROMOTOCO.csv",
                "nifty_stocks/HINDALCO.csv",
                "nifty_stocks/HINDUNILVR.csv",
                "nifty_stocks/ICICIBANK.csv",
                "nifty_stocks/INDUSINDBK.csv",
                "nifty_stocks/INFY.csv",
                "nifty_stocks/ITC.csv",
                "nifty_stocks/JIOFIN.csv",
                "nifty_stocks/JSWSTEEL.csv",
                "nifty_stocks/KOTAKBANK.csv",
                "nifty_stocks/LT.csv",
                "nifty_stocks/M&M.csv",
                "nifty_stocks/MARUTI.csv",
                "nifty_stocks/NESTLEIND.csv",
                "nifty_stocks/NTPC.csv",
                "nifty_stocks/ONGC.csv",
                "nifty_stocks/POWERGRID.csv",
                "nifty_stocks/RELIANCE.csv",
                "nifty_stocks/SBILIFE.csv",
                "nifty_stocks/SBIN.csv",
                "nifty_stocks/SHRIRAMFIN.csv",
                "nifty_stocks/SUNPHARMA.csv",
                "nifty_stocks/TATACONSUM.csv",
                "nifty_stocks/TATAMOTORS.csv",
                "nifty_stocks/TATASTEEL.csv",
                "nifty_stocks/TCS.csv",
                "nifty_stocks/TECHM.csv",
                "nifty_stocks/TITAN.csv",
                "nifty_stocks/TRENT.csv",
                "nifty_stocks/ULTRACEMCO.csv",
                "nifty_stocks/WIPRO.csv"
        ] 
        
        # 2. Load and process data
        print("📊 Loading and processing data...")
        processor = EnhancedFinancialDataProcessor(metadata_path, price_data_paths)
        processor.create_industry_encoder()
        
        features, targets, company_indices = processor.prepare_features_for_all_companies()
        
        # 3. Build financial graph
        print("🕸️ Building financial graph...")
        graph_builder = FinancialGraphBuilder(
            processor.metadata, 
            processor.companies_data, 
            processor.all_price_data
        )
        
        graph_data = graph_builder.build_comprehensive_graph(
            industry_weight=0.4,
            correlation_weight=0.3,
            market_cap_weight=0.2,
            correlation_threshold=0.3
        )
        
        # Visualize the graph
        graph_builder.visualize_graph(graph_data)
        
        # 4. Prepare datasets
        print("📦 Preparing datasets...")
        
        # Time-aware split (80% train, 20% validation)
        split_idx = int(len(features) * 0.8)
        
        train_features = features.iloc[:split_idx]
        val_features = features.iloc[split_idx:]
        train_targets = targets.iloc[:split_idx]
        val_targets = targets.iloc[split_idx:]
        train_company_indices = company_indices[:split_idx]
        val_company_indices = company_indices[split_idx:]
        
        print(f"Training samples: {len(train_features)}")
        print(f"Validation samples: {len(val_features)}")
        
        # Creation of dataset
        train_dataset = EnhancedFinancialDataset(train_features, train_targets, train_company_indices)
        val_dataset = EnhancedFinancialDataset(val_features, val_targets, val_company_indices)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)  # Don't shuffle for time series
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # 5. Initialize the enhanced model
        print("🤖 Initializing Enhanced FinGAT model...")
        
        # Remove company-specific columns from feature count
        input_dim = train_features.shape[1]
        
        model = EnhancedFinancialGAT(
            input_dim=input_dim,
            hidden_dim=256,
            num_companies=processor.num_companies,
            num_heads=8,
            num_layers=3,
            dropout=0.2
        )
        
        print(f"Model created with:")
        print(f"  - Input dimension: {input_dim}")
        print(f"  - Hidden dimension: 256")
        print(f"  - Number of companies: {processor.num_companies}")
        print(f"  - Number of GAT layers: 3")
        print(f"  - Number of attention heads: 8")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 6. Initialize trainer
        trainer = EnhancedFinancialTrainer(model, graph_data)
        
        # 7. Train the model
        print("🎯 Starting training...")
        train_losses, val_losses = trainer.train(
            train_loader, 
            val_loader, 
            epochs=150, 
            lr=0.001
        )
        
        print("✅ Training completed successfully!")
        
        # 8. Load best model and evaluate
        print("📈 Loading best model for final evaluation...")
        
        checkpoint = torch.load('best_enhanced_fingat_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation
        trainer.model = model
        final_val_loss, final_mae, final_acc = trainer.validate(val_loader)
        
        print(f"🏆 Final Results:")
        print(f"  - Validation Loss: {final_val_loss:.4f}")
        print(f"  - Mean Absolute Error: {final_mae:.4f}")
        print(f"  - Direction Accuracy: {final_acc:.4f}")
        
        # 9. Create additional visualizations
        print("📊 Creating additional visualizations...")
        
        # Feature importance visualization
        create_feature_importance_plot(features)
        
        # Performance comparison
        create_performance_comparison_plot(train_losses, val_losses)
        
        # Prediction vs actual scatter plot
        create_prediction_scatter_plot(trainer, val_loader, processor.idx_to_symbol)
        
        print("🎉 Enhanced FinGAT training and evaluation completed!")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


def create_feature_importance_plot(features):
    """Create a feature importance visualization"""
    plt.figure(figsize=(12, 8))
    
    # Calculate feature correlations with each other
    feature_cols = [col for col in features.columns if col not in ['company_idx', 'industry_encoded']]
    feature_corr = features[feature_cols].corr()
    
    # Create heatmap
    sns.heatmap(feature_corr, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature statistics
    plt.figure(figsize=(15, 6))
    features[feature_cols].boxplot(rot=45)
    plt.title('Feature Distributions')
    plt.tight_layout()
    plt.savefig('enhanced_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_performance_comparison_plot(train_losses, val_losses):
    """Create performance comparison plots"""
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8, linewidth=2)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8, linewidth=2)
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Smoothed curves
    plt.subplot(1, 3, 2)
    window = 10
    train_smooth = pd.Series(train_losses).rolling(window).mean()
    val_smooth = pd.Series(val_losses).rolling(window).mean()
    plt.plot(train_smooth, label=f'Train Loss (MA-{window})', alpha=0.8, linewidth=2)
    plt.plot(val_smooth, label=f'Val Loss (MA-{window})', alpha=0.8, linewidth=2)
    plt.title('Smoothed Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate effect
    plt.subplot(1, 3, 3)
    improvement = np.diff(val_losses)
    plt.plot(improvement, alpha=0.7, color='red', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Validation Loss Improvement')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Change')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_prediction_scatter_plot(trainer, val_loader, idx_to_symbol):
    """Create prediction vs actual scatter plot"""
    trainer.model.eval()
    
    all_price_preds = []
    all_price_targets = []
    all_direction_preds = []
    all_direction_targets = []
    company_names = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(trainer.device)
            price_target = batch['price_target'].to(trainer.device)
            direction_target = batch['direction_target'].to(trainer.device)
            company_indices = batch['company_index']
            
            price_pred, direction_pred = trainer.model(features, company_indices, trainer.graph_data)
            
            all_price_preds.extend(price_pred.cpu().numpy())
            all_price_targets.extend(price_target.cpu().numpy())
            all_direction_preds.extend(direction_pred.cpu().numpy())
            all_direction_targets.extend(direction_target.cpu().numpy())
            
            # Get company names
            for idx in company_indices:
                company_names.append(idx_to_symbol.get(idx, f"Company_{idx}"))
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    # Price predictions
    plt.subplot(1, 3, 1)
    plt.scatter(all_price_targets, all_price_preds, alpha=0.6, s=30)
    plt.plot([min(all_price_targets), max(all_price_targets)], 
             [min(all_price_targets), max(all_price_targets)], 'r--', alpha=0.8)
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Price Return Predictions')
    plt.grid(True, alpha=0.3)
    
    # Direction predictions
    plt.subplot(1, 3, 2)
    plt.scatter(all_direction_targets, all_direction_preds, alpha=0.6, s=30)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    plt.xlabel('Actual Direction')
    plt.ylabel('Predicted Direction')
    plt.title('Direction Predictions')
    plt.grid(True, alpha=0.3)
    
    # Residuals
    plt.subplot(1, 3, 3)
    residuals = np.array(all_price_targets) - np.array(all_price_preds)
    plt.scatter(all_price_preds, residuals, alpha=0.6, s=30)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('Predicted Returns')
    plt.ylabel('Residuals')
    plt.title('Prediction Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    mae = mean_absolute_error(all_price_targets, all_price_preds)
    acc = accuracy_score(all_direction_targets, np.round(all_direction_preds))
    
    print(f"📊 Detailed Performance Metrics:")
    print(f"  - Price MAE: {mae:.6f}")
    print(f"  - Direction Accuracy: {acc:.4f}")
    print(f"  - Price R²: {np.corrcoef(all_price_targets, all_price_preds)[0,1]**2:.4f}")


if __name__ == "__main__":
    main()
