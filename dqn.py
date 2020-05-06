from q_learning.utils import get_data
from q_learning.run import run
import utils

if __name__ == '__main__':
    df = utils.get_data()
    weights_dir = './q_learning/weights'
    portfolio_dir = './q_learning/portfolio_val'
    run(df, weights_dir, portfolio_dir)
