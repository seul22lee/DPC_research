import torch
import torch.nn as nn
import torch.optim as optim


def combine_inputs(x_past, y_past, x_future, y_ref, y_const):
    """Combine multiple input sequences along the feature dimension."""
    return torch.cat([x_past, y_past, x_future, y_ref, y_const], dim=-1)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=16, num_layers=2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.mu = nn.Linear(lstm_out_dim, latent_dim)
        self.logvar = nn.Linear(lstm_out_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        mu = self.mu(h_last)
        logvar = self.logvar(h_last)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.fc_latent = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, seq_len=50):
        h0 = torch.tanh(self.fc_latent(z)).unsqueeze(0)
        h0 = h0.repeat(self.lstm.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        dec_in = torch.zeros(z.size(0), seq_len, h0.size(-1), device=z.device)
        out, _ = self.lstm(dec_in, (h0, c0))
        out = self.output_layer(out)
        return out


class SequenceVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=16, num_layers=2):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim, num_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, seq_len=x.size(1))
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1e-3):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae(model, train_data, val_data, num_epochs=20, lr=1e-3, batch_size=256, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss, recon, kl = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon_x, mu, logvar = model(x)
                loss, _, _ = vae_loss(recon_x, x, mu, logvar)
                val_loss += loss.item()

        print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    return model
