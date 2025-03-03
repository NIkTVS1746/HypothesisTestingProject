import pandas as pd  # Импортируем библиотеку pandas для работы с данными в табличном формате
import numpy as np  # Импортируем библиотеку numpy для работы с массивами и числовыми операциями
from enum import Enum  # Импортируем класс Enum для создания перечислений
from datetime import datetime, timedelta  # Импортируем классы для работы с датами и временем
from tqdm import tqdm  # Добавляем импорт tqdm
import matplotlib.pyplot as plt
import seaborn as sns  # Импортируем библиотеку seaborn для построения графиков

def adjust_market_time(base_timestamp, time_offset_minutes):
    
    """
    Корректирует временную метку с учетом смещения рынка и дополнительного времени.
    
    Args:
        base_timestamp: базовая временная метка
        time_offset_minutes: смещение в минутах
    """
    
    market_adjusted_time = base_timestamp - 120*60  # Корректировка на 2 часа
    final_time = market_adjusted_time + time_offset_minutes * 60
    return final_time



def validate_price_prediction(forecast_price, previous_price, actual_price_change, 
                            trading_signal, price_threshold, strategy_type):
    """
    Проверяет корректность прогноза цены на основе выбранной торговой стратегии.
    
    Args:
        forecast_price: прогнозируемая цена
        previous_price: предыдущая цена
        actual_price_change: фактическое изменение цены
        trading_signal: тип сигнала (Long/Short/Neutral)
        price_threshold: пороговое значение изменения цены
        strategy_type: тип стратегии (0: forecast < previous, 1: forecast > previous)
    """
    if strategy_type == 0:
        if (forecast_price < previous_price and 
            trading_signal == "Long" and 
            actual_price_change > price_threshold):
            return True
        elif (forecast_price > previous_price and 
              trading_signal == "Short" and 
              actual_price_change < price_threshold):
            return True
        elif (trading_signal == "Neutral" and 
              actual_price_change == price_threshold):
            return True
    elif strategy_type == 1:
        if (forecast_price > previous_price and 
            trading_signal == "Long" and 
            actual_price_change > price_threshold):
            return True
        elif (forecast_price < previous_price and 
              trading_signal == "Short" and 
              actual_price_change < price_threshold):
            return True
        elif (trading_signal == "Neutral" and 
              actual_price_change == price_threshold):
            return True
    return False




def calculate_price_movement(entry_time, exit_time, entry_date, exit_date):
    """
    Рассчитывает абсолютное и процентное изменение цены между точками входа и выхода.
    """
    price_data = pd.read_csv("BRENT_M5_201304022200_202502051135.csv")
    
    entry_price_data = price_data[
        (price_data["<DATE>"] == entry_date) & 
        (price_data["<TIME>"] == entry_time)
    ]
    exit_price_data = price_data[
        (price_data["<DATE>"] == exit_date) & 
        (price_data["<TIME>"] == exit_time)
    ]
    
    if entry_price_data.empty or exit_price_data.empty:
        return 101, 101, 101, 101 # Индикатор отсутствия данных
        
    entry_price = entry_price_data["<CLOSE>"].values[0]
    exit_price = exit_price_data["<CLOSE>"].values[0]
    
    absolute_change = exit_price - entry_price
    percentage_change = ((exit_price - entry_price) / entry_price) * 100
    
    return absolute_change, percentage_change, entry_price, exit_price



def plot_statistics(data):
    """
    Строит графики для фактических, предыдущих и прогнозируемых значений.
    
    Args:
        actual: список фактических значений
        previous: список предыдущих значений
        forecast: список прогнозируемых значений
    """
    df = pd.DataFrame(data)

    df['date'] = pd.to_datetime(df['date'])

    plt.subplot(1, 3, 1)
    sns.histplot(df['actual_value'], kde=True, color='blue', bins=30)
    plt.title('Распределение фактических значений')
    plt.xlabel('Значения')
    plt.ylabel('Частота')

    plt.subplot(1, 3, 2)
    sns.histplot(df['previous'], kde=True, color='orange', bins=30)
    plt.title('Распределение предыдущих значений')
    plt.xlabel('Значения')
    plt.ylabel('Частота')

    plt.subplot(1, 3, 3)
    sns.histplot(df['forecast'], kde=True, color='green', bins=30)
    plt.title('Распределение прогнозируемых значений')
    plt.xlabel('Значения')
    plt.ylabel('Частота')

    plt.tight_layout()
    plt.show()


def plot_forecast_vs_actual(data):
  
    df = pd.DataFrame(data)

    df['date'] = pd.to_datetime(df['date'])

    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['previous'], label='Previous', marker='o')
    plt.plot(df['date'], df['forecast'], label='Forecast', marker='o')
    plt.plot(df['date'], df['actual_value'], label='Actual', marker='o')

    plt.title('Сравнение Previous, Forecast и Actual по времени')
    plt.xlabel('Время')
    plt.ylabel('Значения')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

 
def generate_trading_parameters(market_events, price_thresholds):
    """
    Генерирует параметры для анализа торговых стратегий.
    """
    strategy_types = [0, 1]  # 0: forecast < previous, 1: forecast > previous
    entry_time_offsets = [60, 120, 5, 30, 600, -5]  # минуты до события
    exit_time_offsets = [60, 120, 5, 30, 600]  # минуты после события
    trading_signals = ["Long", "Short", "Neutral"]
    price_change_thresholds = price_thresholds
    
    return (market_events, entry_time_offsets, exit_time_offsets, 
            trading_signals, price_change_thresholds, strategy_types)


def convert_to_timestamp(datetime_obj):
    """
    Конвертирует datetime объект в Unix timestamp.
    """
    return int(datetime_obj.timestamp())



def analyze_market_events(events_data, price_thresholds):
    """
    Анализирует рыночные события и генерирует торговые сигналы.
    """
    (market_events, entry_time_offsets, exit_time_offsets, 
     trading_signals, price_thresholds, strategy_types) = generate_trading_parameters(
         events_data, price_thresholds)
    
    total_profit = 0.0
    total_loss = 0.0
    successful_trades = []
    hypothesis_counter = -1
    total_hypotheses = 0

    total_iterations = (
        len(strategy_types) *
        len(market_events) *
        len(entry_time_offsets) *
        len(exit_time_offsets) *
        len(trading_signals) *
        len(price_thresholds)
    )

    progress_bar = tqdm(total=total_iterations, 
                       desc="Анализ торговых стратегий",
                       unit="итерация")

    for strategy_type in strategy_types:
        for event in market_events:
            market_open_time = convert_to_timestamp(event['date'])
            
            for entry_offset in entry_time_offsets:
                entry_time = adjust_market_time(market_open_time, -entry_offset)

                for exit_offset in exit_time_offsets:
                    exit_time = adjust_market_time(market_open_time, exit_offset)

                    entry_date = datetime.fromtimestamp(entry_time).strftime("%Y.%m.%d")
                    entry_time_str = datetime.fromtimestamp(entry_time).strftime("%H:%M:%S")
                    exit_date = datetime.fromtimestamp(exit_time).strftime("%Y.%m.%d")
                    exit_time_str = datetime.fromtimestamp(exit_time).strftime("%H:%M:%S")

                    price_change, percentage_change, entry_price, exit_price = calculate_price_movement(
                        entry_time_str, exit_time_str, entry_date, exit_date
                    )
                    
                    if price_change == 101:
                        progress_bar.update(len(trading_signals) * len(price_thresholds))
                        continue

                    for signal in trading_signals:
                        for threshold in price_thresholds:
                            total_hypotheses += 1

                            is_successful = validate_price_prediction(
                                event["forecast"],
                                event["previous"],
                                percentage_change,
                                signal,
                                threshold,
                                strategy_type
                            )
                            
                            trade_info = {
                                "hypothesis_id": hypothesis_counter + 1,
                                "event": event["title"],
                                "entry_date": entry_date,
                                "entry_time": entry_time_str,
                                "exit_date": exit_date,
                                "exit_time": exit_time_str,
                                "signal_type": signal,
                                "strategy_type": 'forecast<previous' if strategy_type == 0 else 'forecast>previous',
                                "is_successful": is_successful,
                                "price_threshold": threshold,
                                "price_change": f"+{percentage_change}%" if percentage_change > 0 else f"{percentage_change}%",
                            }
                            #"price_change": f"+{percentage_change:.2f}%" if percentage_change > 0 else f"{percentage_change:.2f}%
                            if is_successful:
                                total_profit += abs(percentage_change)
                                hypothesis_counter += 1
                                successful_trades.append(trade_info)

                                # Вывод информации о успешной сделке
                                print(f"Успешная сделка: "
                                      f"Входная дата: {entry_date}, "
                                      f"Входное время: {entry_time_str}, "
                                      f"Выходная дата: {exit_date}, "
                                      f"Выходное время: {exit_time_str}, "
                                      f"Цена открытия: {entry_price}, "
                                      f"Цена закрытия: {exit_price}")

                            else:
                                total_loss += abs(percentage_change)
                            
                            progress_bar.update(1)

    progress_bar.close()

    success_rate = (len(successful_trades)/total_hypotheses) * 100 
    net_result = total_profit - total_loss
    
    statistics = {
        "total_hypotheses": total_hypotheses,
        "successful_trades": len(successful_trades),
        "success_rate": success_rate,
        "total_profit": total_profit,
        "total_loss": total_loss,
        "net_result": f"+{net_result}" if net_result > 0 else f"{net_result}"
    }
    plot_forecast_vs_actual(events_data)
    plot_statistics(events_data)
    return successful_trades, statistics



""" 
df = pd.DataFrame(trades)
df.to_csv('df.csv', index=False)
print(trades)
print(df) """

""" def calculate_statistics(actual, previous, forecast):
    statistics = {
        'variance_actual': np.var(actual),
        'std_dev_actual': np.std(actual),
        'variance_previous': np.var(previous),
        'std_dev_previous': np.std(previous),
        'variance_forecast': np.var(forecast),
        'std_dev_forecast': np.std(forecast),
        'correlation': np.corrcoef(actual, forecast)[0, 1]  # Корреляция между actual и forecast
    }
    return statistics """




