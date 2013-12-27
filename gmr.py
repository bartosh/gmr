"""
Adapted Global Market Rotation Strategy

This strategy rotates between six global market ETFs on a monthly
basis.  Each month the performance and mean 20-day volitility over
the last 13 weekds are used to rank which ETF should be invested
in for the coming month.

"""

import math


def initialize(context):
    """Initialize context object. It's passed to the handle_data function."""
    context.stocks = [
        sid(12915), # MDY (SPDR S&P MIDCAP 400)
        sid(21769), # IEV (ISHARES EUROPE ETF)
        sid(24705), # EEM (ISHARES MSCI EMERGING MARKETS)
        sid(23134), # ILF (ISHARES LATIN AMERICA 40)
        sid(23118), # EEP (ISHARES MSCI PACIFIC EX JAPAN)
        sid(22887), # EDV (VANGUARD EXTENDED DURATION TREASURY)
    ]

    # Date of last rebalancing
    context.rebalancing_date = None

    # Rebalancing period in calendar days
    context.period = 31

    # The order ID of the sell order currently being filled
    context.oid = None

    # The current stock being held
    context.currentStock = None

    # The next stock that needs to get purchased (once the sell order
    # on the current stock is filled
    context.nextStock = None

    # The 3-month lookback period.  Calculated based on there being
    # an average of 21 trading days in a month
    context.lookback = 63

def getminmax(vdict):
    """
    Get the minimum and maximum values of a list of dictionary values.

    :param vdict: Python dict-like object.
    :returns: minimum and maximum of vdict values
    """
    vals = vdict.values()
    return min(vals), max(vals)

def hist_volatility(period, prices):
    """
    Calculate the n-day historical volatility given a set of n+1 prices.

    :param period: The number of days for which to calculate volatility
    :param prices: An array of price information.  Must be of length period + 1.
    """
    # HVdaily = sqrt( sum[1..n](x_t - Xbar)^2 / n - 1)

    # Start by calculating Xbar = 1/n sum[1..n] (ln(P_t / P_t-1))
    returns = []
    for i in xrange(1, period + 1):
        returns.append(math.log(prices[i] / prices[i-1]))

    # Find the average of all returns
    rmean = sum(returns) / period

    # Determine the difference of each return from the mean, then square
    diff = []
    for i in xrange(0, period):
        diff.append(math.pow((returns[i] - rmean), 2))

    # Take the square root of the sum over the period - 1.  Then mulitply
    # that by the square root of the number of trading days in a year
    vol = math.sqrt(sum(diff) / (period - 1)) * math.sqrt(252/period)

    return vol

def getmetrics(prices, period):
    """
    Get the performance and average 20-day volatility of a security
    over a given period

    :param prices:
    :param period: The time period for which to find
    """
    # Get the prices
    #prices = data['close_price'][security][-period-1:]
    start = prices[-period] # First item
    end = prices[-1] # Last item

    performance = (end - start) / start

    # Calculate 20-day volatility for the given period
    volats = []
    j = 0
    for i in xrange(-period, 0):
        volats.append(hist_volatility(20, prices[i-21:21+j]))
        j += 1

    avg_volat = sum(volats) / period

    return performance, avg_volat

def getbeststock(data, stocks, period):
    """
    Pick the best stock from a group of stocks based on the given
    data over a specified period using the stocks' performance and
    volatility

    :param data: The datapanel with data of all the stocks
    :param stocks: A list of stocks to rank
    :param period: The time period over which the stocks will be analyzed
    """
    best = None

    performances = {}
    volatilities = {}

    # Get performance and volatility for all the stocks
    for stock in stocks:
        perf, volat = getmetrics(data['price'][stock.sid], period)
        performances[stock.sid] = perf
        volatilities[stock.sid] = volat

    # Determine min/max of each.  NOTE: volatility is switched
    # since a low volatility should be weighted highly.
    minp, maxp = getminmax(performances)
    maxv, minv = getminmax(volatilities)

    # Normalize the performance and volatility values to a range
    # between [0..1] then rank them based on a 70/30 weighting.
    for stock in stocks:
        perf = (performances[stock.sid] - minp) / (maxp - minp)
        volat = (volatilities[stock.sid] - minv) / (maxv - minv)
        rank = perf * 0.7 + volat * 0.3

        #log.info('Rank info for %s: p=%s, v=%s, r=%s' % (s,p,v,rank))

        # If the new rank is greater than the old best rank, pick it.
        if best is None or rank > best[1]:
            best = stock, rank

    return best[0]

def sellholdings(context):
    """Sell all the currently held positions in the context's portfolio."""
    positions = context.portfolio.positions

    oid = None
    for pos in positions.values():
        if (pos.amount > 0):
            log.info('Selling %s shares of %s' % (pos.amount, pos.sid.symbol))
            oid = order(pos.sid, -pos.amount)

    return oid

@batch_transform(window_length=83)
def accumulatedata(data):
    """
    Utilize the batch_transform decorator to accumulate multiple days
    of data into one datapanel  Need the window length to be 20 longer
    than lookback period to allow for a 20-day volatility calculation
    """
    return data

def days(begin, end):
    """Calculate amount of calendar days between two dates."""
    roundb = begin.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    rounde = end.replace(hour = 0, minute = 0, second = 0, microsecond = 0)

    return (rounde - roundb).days

def handle_data(context, data):
    """
    The main proccessing function.
    Called whenever a market event occurs for any of algorithm's securities.

    :param context: context object
    :param data: Object contains all the market data for algorithm securities
                 keyed by security id. It represents a snapshot of algorithm's
                 universe as of when this method is called.
    :returns: None
    """
    # Accumulate data until there is enough days worth of data
    # to process without having outOfBounds issues.
    datapanel = accumulatedata(data)

    if datapanel is None:
        # There is insufficient data accumulated to process
        return

    current_date = get_datetime()

    # If there is an order ID, check the status of the order.
    # If there is an order and it is filled, the next stock
    # can be purchased.
    if context.oid is not None:
        orderobj = get_order(context.oid)
        if orderobj.filled == orderobj.amount:
            log.info('Sold %s shares of %s' % (-orderobj.amount, orderobj.sid.symbol))
            # Good to buy next holding
            cash = context.portfolio.cash
            oobj = get_order(order_value(context.nextStock, cash))
            log.info('Sell order complete, buying %s shares of %s. Cash is %s' % \
                 (oobj.amount, context.nextStock.symbol, cash))
            context.currentStock = context.nextStock
            context.oid = None
            context.nextStock = None
            context.rebalancing_date = current_date

    if context.rebalancing_date and days(context.rebalancing_date,
                                         current_date) < context.period:
        # It's not a time to rebalance yet, nothing further to do
        return

    # At this point, a new month has been reached.  The stocks
    # need to be

    # Ensure stocks are only traded if possible.
    # (e.g) EDV doesn't start trading until late 2007, without
    # this, any backtest run before that date would fail.
    stocks = []
    for stock in context.stocks:
        if current_date > stock.security_start_date \
                 and stock.sid in datapanel['price']:
            stocks.append(stock)

    # Determine which stock should be used for the next month
    best = getbeststock(datapanel, stocks, context.lookback)

    if best:
        if (context.currentStock is not None and context.currentStock == best):
            # If there is a stock currently held and it is the same as
            # the new 'best' stock, nothing needs to be done
            return
        else:
            # Otherwise, the current stock needs to be sold and the new
            # stock bought
            context.oid = sellholdings(context)
            context.nextStock = best

            # Purchase will not occur until the next call of handle_data
            # and only when the order has been filled.

    # If there is no stock currently held, it needs to be bought.
    # This only happend
    if context.currentStock is None:
        cash = context.portfolio.cash
        oobj = get_order(order_value(context.nextStock, cash))
        log.info('Buying %s shares of %s. Cash is %s' % \
                 (oobj.amount, context.nextStock.symbol, cash))
        context.currentStock = context.nextStock
        context.oid = None
        context.nextStock = None
        context.rebalancing_date = current_date
