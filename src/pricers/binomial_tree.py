import numpy as np
import matplotlib.pyplot as plt

def price_gbm_put(
    asset0: float,      
    sigma: float,       
    r: float,           
    strike: float,      
    t: float,           
    cnt_times: int      
):
    def _generate_tree():
        dt = t / (cnt_times - 1)

        tree = np.zeros((cnt_times, cnt_times), dtype=float)

        det_comp = (r - 0.5 * sigma * sigma) * dt  
        stoc_comp = sigma * np.sqrt(dt)

        up = np.exp(det_comp + stoc_comp)
        down = np.exp(det_comp - stoc_comp)

        f = np.vectorize(
            lambda i, j: up**(j - i) * down**i if i <= j else 0
        )
        tree = asset0 * np.fromfunction(f, (cnt_times, cnt_times))

        return dt, up, down, tree

    dt, up, down, tree = _generate_tree()

    def _plot(tree):
        plt.figure(figsize=(10, 6))
        
        for i in range(cnt_times):
            for j in range(i + 1):
                plt.plot(dt * i, tree[j, i], 'bo')
        
        label_added = False
        for i in range(cnt_times - 1):
            for j in range(i + 1):
                if not label_added:
                    plt.plot([dt * i, dt * (i + 1)], [tree[j, i], tree[j, i + 1]], 'b-', alpha=0.5, label="tree")
                    label_added = True
                else:
                    plt.plot([dt * i, dt * (i + 1)], [tree[j, i], tree[j, i + 1]], 'b-', alpha=0.5)
                plt.plot([dt * i, dt * (i + 1)], [tree[j, i], tree[j + 1, i + 1]], 'b-', alpha=0.5)

        plt.plot([0, t], [strike, strike], "--", color="red", label="strike")
        
        plt.legend()
        plt.title("Дерево цен актива")
        plt.xlabel("Шаги времени")
        plt.ylabel("Цена актива")
        plt.grid()
        plt.show()

    # _plot(tree)
        
    def _price(payoff_function):
        p_tilde = (np.exp(r * dt) - down) / (up - down)  
        price = np.zeros((cnt_times, cnt_times))
        price[:, -1] = payoff_function(tree[:, -1])

        price_history = np.zeros(cnt_times)
        price_history[-1] = price[:, -1].mean()

        payoff = payoff_function(tree)
        for time_index in range(cnt_times - 2, -1, -1):
            for traj_index in range(time_index + 1):
                price[traj_index, time_index] = np.maximum(
                    payoff[traj_index, time_index],
                    np.exp(-r * dt) * (  
                        p_tilde * price[traj_index, time_index + 1] + 
                        (1 - p_tilde) * price[traj_index + 1, time_index + 1]
                    )
                )
            
            price_history[time_index] = price[:, time_index][
                tree[:, time_index] != 0.
            ].mean()

        return price[0, 0], price_history
    
    return _price(
        lambda x: np.maximum(0, strike - x)
    )