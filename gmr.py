#
# Adapted Global Market Rotation Strategy
#
# This strategy rotates between six global market ETFs on a monthly
# basis.  Each month the performance and mean 20-day volitility over
# the last 13 weekds are used to rank which ETF should be invested
# in for the coming month.

import math
import pandas


def initialize(context):
    context.stocks = {
        12915: sid(12915), # MDY (SPDR S&P MIDCAP 400)
        21769: sid(21769), # IEV (ISHARES EUROPE ETF)
        24705: sid(24705), # EEM (ISHARES MSCI EMERGING MARKETS)
        23134: sid(23134), # ILF (ISHARES LATIN AMERICA 40)
        23118: sid(23118), # EEP (ISHARES MSCI PACIFIC EX JAPAN)
        22887: sid(22887), # EDV (VANGUARD EXTENDED DURATION TREASURY)
        #23911: sid(23911)
    }

    # Keep track of the current month.
    context.currentMonth = None

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

'''
  Gets the minimum and maximum values of an array of values
'''
def getMinMax(arr):
   return min(arr.values()), max(arr.values())

'''
  Calculates the n-day historical volatility given a set of
  n+1 prices.

  @param period The number of days for which to calculate volatility
  @param prices An array of price information.  Must be of length
    period + 1.
'''
def historicalVolatility(period, prices):
    # HVdaily = sqrt( sum[1..n](x_t - Xbar)^2 / n - 1)

    # Start by calculating Xbar = 1/n sum[1..n] (ln(P_t / P_t-1))
    r = []
    for i in xrange(1, period + 1):
        r.append(math.log(prices[i] / prices[i-1]))

    # Find the average of all returns
    rMean = sum(r) / period;

    # Determine the difference of each return from the mean, then square
    d = []
    for i in xrange(0, period):
        d.append(math.pow((r[i] - rMean), 2))

    # Take the square root of the sum over the period - 1.  Then mulitply
    # that by the square root of the number of trading days in a year
    vol = math.sqrt(sum(d) / (period - 1)) * math.sqrt(252/period)

    return vol

'''
  Gets the performance and average 20-day volatility of a security
  over a given period

  @param prices
  @param period The time period for which to find
'''
def getStockMetrics(prices, period):
    # Get the prices
    #prices = data['close_price'][security][-period-1:]
    start = prices[-period] # First item
    end = prices[-1] # Last item

    performance = (end - start) / start

    # Calculate 20-day volatility for the given period
    v = []
    x = 0
    for i in xrange(-period, 0):
        v.append(historicalVolatility(20, prices[i-21:21+x]))
        x += 1

    volatility = sum(v) / period

    return performance, volatility

'''
  Picks the best stock from a group of stocks based on the given
  data over a specified period using the stocks' performance and
  volatility

  @param data The datapanel with data of all the stocks
  @param stocks A list of stocks to rank
  @param period The time period over which the stocks will be
    analyzed
'''
def getBestStock(data, stocks, period):
    best = None

    performances = {}
    volatilities = {}

    # Get performance and volatility for all the stocks
    for s in stocks:
        p, v = getStockMetrics(data['price'][s.sid], period)
        performances[s.sid] = p
        volatilities[s.sid] = v

    # Determine min/max of each.  NOTE: volatility is switched
    # since a low volatility should be weighted highly.
    minP, maxP = getMinMax(performances)
    maxV, minV = getMinMax(volatilities)

    # Normalize the performance and volatility values to a range
    # between [0..1] then rank them based on a 70/30 weighting.
    for s in stocks:
        p = (performances[s.sid] - minP) / (maxP - minP)
        v = (volatilities[s.sid] - minV) / (maxV - minV)
        rank = p * 0.7 + v * 0.3

        #log.info('Rank info for %s: p=%s, v=%s, r=%s' % (s,p,v,rank))

        # If the new rank is greater than the old best rank, pick it.
        if best is None or rank > best[1]:
            best = s, rank

    return best[0]

'''
  Sells all the currently held positions in the context's portfolio
'''
def sellHoldings(context):
    positions = context.portfolio.positions

    oid = None
    for p in positions.values():
        if (p.amount > 0):
            #log.debug('ordering %s' % p)
            oid = order(p.sid, -p.amount)

    return oid

'''
  Utilize the batch_transform decorator to accumulate multiple days
  of data into one datapanel  Need the window length to be 20 longer
  than lookback period to allow for a 20-day volatility calculation
'''
@batch_transform(window_length=83)
def accumulateData(data):
    return data


'''
  The main proccessing function.  This is called and passed data
'''
def handle_data(context, data):
    # Accumulate data until there is enough days worth of data
    # to process without having outOfBounds issues.
    datapanel = accumulateData(data)

    if datapanel is None:
        # There is insufficient data accumulated to process
        return


    # If there is an order ID, check the status of the order.
    # If there is an order and it is filled, the next stock
    # can be purchased.
    if context.oid is not None:
        orderObj = get_order(context.oid)
        if orderObj.filled == orderObj.amount:
            # Good to buy next holding
            amount = math.floor((context.portfolio.cash) / data[context.nextStock.sid].price) - 1
            log.info('Sell order complete, buying %s of %s (%s of %s)' % \
                 (amount, context.nextStock, amount*data[context.nextStock.sid].price, context.portfolio.cash))
            order(context.nextStock, amount)
            context.currentStock = context.nextStock
            context.oid = None
            context.nextStock = None


    date = get_datetime()
    month = date.month

    if not context.currentMonth:
        # Set the month initially
        context.currentMonth = month

    if context.currentMonth == month:
        # If the current month is unchanged, nothing further to do
        return

    context.currentMonth = month

    # At this point, a new month has been reached.  The stocks
    # need to be

    # Ensure stocks are only traded if possible.
    # (e.g) EDV doesn't start trading until late 2007, without
    # this, any backtest run before that date would fail.
    stocks = []
    for s in context.stocks.values():
        if date > s.security_start_date:
            stocks.append(s)


    # Determine which stock should be used for the next month
    best = getBestStock(datapanel, stocks, context.lookback)

    if best:
        if (context.currentStock is not None and context.currentStock == best):
            # If there is a stock currently held and it is the same as
            # the new 'best' stock, nothing needs to be done
            return
        else:
            # Otherwise, the current stock needs to be sold and the new
            # stock bought
            context.oid = sellHoldings(context)
            context.nextStock = best

            # Purchase will not occur until the next call of handle_data
            # and only when the order has been filled.

    # If there is no stock currently held, it needs to be bought.
    # This only happend
    if context.currentStock is None:
        amount = math.floor((context.portfolio.cash) / data[context.nextStock.sid].price) - 1
        log.info('First purchase, buying %s of %s (%s of %s)' % \
             (amount, context.nextStock, amount*data[context.nextStock.sid].price, context.portfolio.cash))
        order(context.nextStock, amount)
        context.currentStock = context.nextStock
        context.oid = None
        context.nextStock = None
