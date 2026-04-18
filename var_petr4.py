import subprocess
import sys

# Ensure dependencies are installed
def install_dependencies():
    deps = ['yfinance', 'numpy', 'scipy', 'matplotlib', 'pandas']
    try:
        import yfinance
        import scipy
        import matplotlib
        import pandas
        import numpy
    except ImportError:
        print("Instalando dependências ausentes...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + deps)

install_dependencies()

import yfinance as yf
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import datetime

# Constantes e Parâmetros
TICKER = "PETR4.SA"
PORTFOLIO_VALUE = 100000.0
ALPHA = 0.05
HORIZON_1D = 1
HORIZON_10D = 10
YEARS = 2

# Seed para reprodutibilidade no Monte Carlo
np.random.seed(42)

def download_data(ticker=TICKER, years=YEARS):
    """
    Baixa os dados históricos de preços usando yfinance
    e calcula os log-retornos diários.
    """
    try:
        # Define a janela de tempo (últimos 2 anos)
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=years*365)
        
        print(f"Baixando dados reais para {ticker} ({start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')})...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            raise ValueError("Nenhum dado retornado. O Ticker pode estar incorreto.")
            
        # O yfinance mais recente pode retornar um MultiIndex no topo das colunas se múltiplos tickers
        # forem passados, ou mesmo para 1 ticker.
        if isinstance(df.columns, pd.MultiIndex):
            adj_close = df['Adj Close'][ticker]
        elif 'Adj Close' in df.columns:
            adj_close = df['Adj Close']
        elif 'Close' in df.columns: 
            adj_close = df['Close']
        else:
            raise KeyError("Coluna 'Adj Close' ou 'Close' não encontrada.")
            
        # Limpa dias sem negociação
        adj_close = adj_close.dropna()
        
        # Log-retornos diários: ln(P_t / P_{t-1})
        log_returns = np.log(adj_close / adj_close.shift(1)).dropna()
        
        return adj_close, log_returns
    except Exception as e:
        print(f"Falha ao baixar dados: {e}")
        sys.exit(1)

def calc_var_param(returns, portfolio_value, alpha):
    """
    Calcula o VaR e CVaR usando o Método Paramétrico (Analítico Normal).
    """
    # Passo 1: Calcula μ (média) e σ (desvio padrão) dos log-retornos
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    
    # Passo 2: Use scipy.stats.norm.ppf(alpha) para obter o z-score
    # Como alpha=0.05, z será negativo (aprox -1.645)
    z = stats.norm.ppf(alpha)
    
    # Passo 3: VaR = portfolio_value * (-z * σ - μ)
    var = portfolio_value * (-z * sigma - mu)
    
    # Passo 4: CVaR = portfolio_value * (σ * norm.pdf(z) / alpha - μ)
    # A norm.pdf nos dá o valor da densidade de probabilidade no z-score
    pdf_z = stats.norm.pdf(z)
    cvar = portfolio_value * (sigma * pdf_z / alpha - mu)
    
    return var, cvar

def calc_var_hist(returns, portfolio_value, alpha):
    """
    Calcula o VaR e CVaR usando o Método de Simulação Histórica.
    """
    # Passo 1: Ordene os retornos do pior para o melhor com np.sort()
    sorted_returns = np.sort(returns.values)
    n = len(sorted_returns)
    
    # Passo 2: O VaR é o retorno na posição int(alpha * n)
    cut_idx = int(alpha * n)
    retorno_no_corte = sorted_returns[cut_idx]
    
    # Passo 3: VaR = portfolio_value * (-retorno_no_corte)
    var = portfolio_value * (-retorno_no_corte)
    
    # Passo 4: O CVaR é a média dos retornos abaixo desse limiar
    # Selecionamos todos os componentes piores que o nosso limite estipulado no VaR
    piores_retornos = sorted_returns[:cut_idx]
    cvar = portfolio_value * (-np.mean(piores_retornos))
    
    return var, cvar, retorno_no_corte

def calc_var_mc(returns, portfolio_value, alpha, n_simulations=50000):
    """
    Calcula o VaR e CVaR usando o Método de Monte Carlo.
    """
    # Passo 1: Use os parâmetros históricos (μ, σ) como inputs
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    
    # Passo 2: Simule 50.000 retornos diários com np.random.normal(μ, σ, 50_000)
    simulated_returns = np.random.normal(mu, sigma, n_simulations)
    
    # Passo 3: Ordene e aplique o mesmo corte percentual do método histórico
    sorted_sims = np.sort(simulated_returns)
    cut_idx = int(alpha * n_simulations)
    retorno_no_corte = sorted_sims[cut_idx]
    
    # Passo 4: Retorne o VaR e a média abaixo do limiar para CVaR
    var = portfolio_value * (-retorno_no_corte)
    piores_retornos_sims = sorted_sims[:cut_idx]
    cvar = portfolio_value * (-np.mean(piores_retornos_sims))
    
    return var, cvar

def kupiec_test(returns, var_1d, portfolio_value, alpha):
    """
    Extensão Opcional: Teste de Kupiec (Backtesting do VaR).
    O teste verifica se a proporção de violações (perdas reais que superam 
    o VaR previsto) é estatisticamente consistente com o nível de confiança (alpha).
    """
    # Converte var_1d em percentual para ser comparado diretamente contra o retorno
    var_percentual = var_1d / portfolio_value
    
    # Conta os dias nos quais a perda percentual foi maior do que o VaR projetado
    # Uma perda grande equivale a um log-retorno negativo abaixo do -var_percentual
    violations = np.sum(returns < -var_percentual)
    n = len(returns)
    
    expected_rate = alpha
    actual_rate = violations / n
    expected_violations = expected_rate * n
    
    # Calcula a Razão de Verossimilhança (Likelihood Ratio POF test)
    # Evita erros de divisão por 0 caso violations seja 0
    if violations == 0:
        lr_stat = -2 * np.log((1 - expected_rate)**n)
    elif violations == n:
        lr_stat = -2 * np.log(expected_rate**n)
    else:
        # LR = 2*ln( (p_obs^x * (1-p_obs)^(N-x)) / (p_exp^x * (1-p_exp)^(N-x)) )
        # Podemos escrever de ambas as formas, a fórmula log é mais robusta no Python:
        num = (expected_rate**violations) * ((1 - expected_rate)**(n - violations))
        den = (actual_rate**violations) * ((1 - actual_rate)**(n - violations))
        lr_stat = -2 * np.log(num / den)
            
    # Usa distribuição Chi-quadrado (df=1) para ver se estouramos nosso nível de confiança 95%
    critical_value = stats.chi2.ppf(1 - alpha, df=1) # 95% conf, alpha=0.05
    
    # Se a estatística do modelo (lr_stat) for menor do que o crítico, a hipótese Nula
    # (proporção verdadeira = alpha) não pode ser rejeitada. Nós PASSAMOS no teste.
    passes = lr_stat < critical_value
        
    return int(violations), expected_violations, passes, lr_stat, critical_value

def plot_results(adj_close, returns, var_param, var_hist, var_mc, cvar_param, cvar_hist, cvar_mc, portfolio_value):
    """
    Gera um painel com 4 gráficos matplotlib com estilo quantitativo.
    """
    # Prepara a figura (2 colunas, 2 linhas)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Painel de Gestão de Risco - {TICKER}', fontsize=18, fontweight='bold', y=0.96)

    # Cores
    color_param = '#FF6347'  # Tomato
    color_hist = '#4682B4'   # SteelBlue
    color_mc = '#3CB371'     # MediumSeaGreen

    # =====================================================================
    # [0,0] Histórico de preços com linha de média móvel 20d
    # =====================================================================
    ax00 = axes[0, 0]
    # Linha base
    ax00.plot(adj_close.index, adj_close.values, label='Fechamento Ajustado', color='#2F4F4F', linewidth=1)
    # Média Móvel
    mma_20 = adj_close.rolling(window=20).mean()
    ax00.plot(adj_close.index, mma_20.values, label='Média Móvel (20d)', color='darkorange', linewidth=1.5)
    
    ax00.set_title('Histórico de Preços')
    ax00.set_ylabel('Preço P$')
    ax00.grid(visible=True, alpha=0.3)
    ax00.legend(loc='upper left')
    
    # Rotaciona rótulos do eixo temporal caso necessário
    for tick in ax00.get_xticklabels():
        tick.set_rotation(30)

    # =====================================================================
    # [0,1] Distribuição de retornos com curva normal e VaR Paramétrico
    # =====================================================================
    ax01 = axes[0, 1]
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    
    # Retornos observados em um histograma densidade (para alinhar com curva PDF)
    ax01.hist(returns, bins=50, density=True, alpha=0.5, color='gray', edgecolor='black', label='Retornos puros')
    
    # Curva Normal (Teórica)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    pdf = stats.norm.pdf(x, mu, sigma)
    ax01.plot(x, pdf, color='black', linewidth=2, label='Curva Normal')
    
    # Linha do VaR Paramétrico
    ret_var_param = -(var_param / portfolio_value)
    ax01.axvline(ret_var_param, color=color_param, linestyle='--', linewidth=2, 
                 label=f'VaR Param. ({ret_var_param:.2%})')
    
    ax01.set_title('Distribuição e Normalidade dos Retornos')
    ax01.set_xlabel('Log-Retornos')
    ax01.set_ylabel('Densidade')
    ax01.legend(loc='best')
    ax01.grid(visible=True, alpha=0.3)

    # =====================================================================
    # [1,0] Histograma histórico com linha do VaR Histórico
    # =====================================================================
    ax10 = axes[1, 0]
    
    # Retornos observados
    ax10.hist(returns, bins=50, alpha=0.7, color=color_hist, edgecolor='black', label='Frequência de Retornos')
    
    # Linha do VaR Histórico
    ret_var_hist = -(var_hist / portfolio_value)
    ax10.axvline(ret_var_hist, color='darkred', linestyle='--', linewidth=2, 
                 label=f'VaR Histórico ({ret_var_hist:.2%})')
    
    ax10.set_title('Método Histórico de Retornos Empíricos')
    ax10.set_xlabel('Log-Retornos')
    ax10.set_ylabel('Dias (Frequência)')
    ax10.legend(loc='best')
    ax10.grid(visible=True, alpha=0.3)

    # =====================================================================
    # [1,1] Barras comparativas: VaR e CVaR (3 métodos)
    # =====================================================================
    ax11 = axes[1, 1]
    
    labels = ['Paramétrico', 'Histórico', 'Monte Carlo']
    var_bars = [var_param, var_hist, var_mc]
    cvar_bars = [cvar_param, cvar_hist, cvar_mc]
    
    x_pos = np.arange(len(labels))
    width = 0.35
    
    # Gráfico de barras agrupadas
    bars1 = ax11.bar(x_pos - width/2, var_bars, width, label='VaR (95%)', color='#E6BBAD', edgecolor='black')
    bars2 = ax11.bar(x_pos + width/2, cvar_bars, width, label='CVaR (95%)', color='#B0306A', edgecolor='black')
    
    ax11.set_title('Comparativo VaR x CVaR (Horizonte = 1D)')
    ax11.set_xticks(x_pos)
    ax11.set_xticklabels(labels)
    ax11.set_ylabel('Valor sob Risco(R$)')
    ax11.grid(visible=True, axis='y', alpha=0.3)
    ax11.legend(loc='upper left')
    
    # Colocar o valor numérico acima das barras
    for bar in bars1:
        yval = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width()/2, yval + 150, f'R${yval:,.0f}', ha='center', va='bottom', fontsize=9)
        
    for bar in bars2:
        yval = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width()/2, yval + 150, f'R${yval:,.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout(pad=2.0)
    return fig

def main():
    print("\n" + "=" * 70)
    print(" " * 15 + "VALUATION E RISCO - CÁLCULO DE VaR & CVaR")
    print("=" * 70)
    
    # Fase 1: Aquisição de Dados
    adj_close, returns = download_data(TICKER, YEARS)
    
    # Fase 2: Estatísticas Descritivas
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    worst_day = returns.min()
    best_day = returns.max()
    start_str = adj_close.index[0].strftime('%Y-%m-%d')
    end_str = adj_close.index[-1].strftime('%Y-%m-%d')
    
    print("\n[1] ESTATÍSTICAS DESCRITIVAS DA SÉRIE HISTÓRICA")
    print("-" * 50)
    print(f" Ativo              : {TICKER}")
    print(f" Horizonte temporal : {start_str} até {end_str}")
    print(f" Amostra (dias)     : {len(returns)} pregões")
    print(f" Média dos retornos : {mu:.6%}")
    print(f" Desvio Padrão      : {sigma:.6%}")
    print(f" Pior Dia           : {worst_day:.2%}")
    print(f" Melhor Dia         : {best_day:.2%}")
    
    # Fase 3: Cálculo do Risco
    # 1. Paramétrico
    var_p, cvar_p = calc_var_param(returns, PORTFOLIO_VALUE, ALPHA)
    # 2. Histórico
    var_h, cvar_h, _ = calc_var_hist(returns, PORTFOLIO_VALUE, ALPHA)
    # 3. Monte Carlo
    var_mc, cvar_mc = calc_var_mc(returns, PORTFOLIO_VALUE, ALPHA)
    
    # Escalabilidade temporal do VaR: raiz quadrada do tempo
    sqrt_10 = np.sqrt(HORIZON_10D)
    
    # Imprimindo Resultados 
    print("\n[2] RESULTADOS: VALOR EM RISCO E EXPECTED SHORTFALL")
    print("-" * 75)
    print(f"{'MÉTODO':<18} | {'VaR (1 Dia)':<15} | {'VaR (10 Dias)':<15} | {'CVaR (1 Dia)':<15}")
    print("-" * 75)
    print(f"{'1. Paramétrico':<18} | R$ {var_p:>10.2f} | R$ {var_p * sqrt_10:>10.2f} | R$ {cvar_p:>10.2f}")
    print(f"{'2. Histórico':<18} | R$ {var_h:>10.2f} | R$ {var_h * sqrt_10:>10.2f} | R$ {cvar_h:>10.2f}")
    print(f"{'3. Monte Carlo':<18} | R$ {var_mc:>10.2f} | R$ {var_mc * sqrt_10:>10.2f} | R$ {cvar_mc:>10.2f}")
    print("-" * 75)
    
    # Fase 4: Backtesting do modelo de VaR Paramétrico (Kupiec POF Test)
    print("\n[3] EXTENSÃO: BACKTESTING (TESTE POF DE KUPIEC) - VAR HISTÓRICO")
    print("-" * 50)
    # Vamos rodar no Var Histórico como pedido genérico, ou podemos no Paramétrico.
    # O user pediu para rodar sem especificar, eu passei var_h como a medida de risco projetada:
    viol_obs, viol_exp, passes, lr_stat, crit_val = kupiec_test(returns, var_h, PORTFOLIO_VALUE, ALPHA)
    
    print(f" -> Violações Esperadas (5% de N) : {viol_exp:.1f} dias")
    print(f" -> Violações Observadas reais    : {viol_obs} dias")
    print(f" -> Test Statistic LR_pof         : {lr_stat:.4f} (Crítico: {crit_val:.4f})")
    
    status = "APROVADO (Modelo Adequado)" if passes else "REJEITADO (Subestimou/Superestimou o risco)"
    print(f" -> STATUS                        : {status}")
    
    # Fase 5: Visualização
    print("\n>> Gerando o Painel de Controle Visual. Por favor, visualize a janela popup.")
    fig = plot_results(adj_close, returns, var_p, var_h, var_mc, cvar_p, cvar_h, cvar_mc, PORTFOLIO_VALUE)
    
    # Exibe
    plt.show()

if __name__ == "__main__":
    # Se quiser ignorar futurewarnings do pandas atrelados ao yfinance
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    main()
