import argparse
import random

from client.exchange_service.client import BaseExchangeServerClient
from protos.order_book_pb2 import Order
from protos.service_pb2 import PlaceOrderResponse

BID_PERCENTILE = 0.05
ASK_PERCENTILE = 1-BID_PERCENTILE
QUANTITY = 1

class ExampleMarketMaker(BaseExchangeServerClient):

    def __init__(self, *args, **kwargs):
        BaseExchangeServerClient.__init__(self, *args, **kwargs)

        self._bids = {"C98PHX": set(), "C99PHX": set(), "C100PHX": set(), "C101PHX": set(),
                      "C102PHX": set(), "P98PHX": set(), "P99PHX": set(), "P100PHX": set(),
                      "P101PHX": set(), "P102PHX": set(), "IDX#PHX": set()}
        self._asks = {"C98PHX": set(), "C99PHX": set(), "C100PHX": set(), "C101PHX": set(),
                      "C102PHX": set(), "P98PHX": set(), "P99PHX": set(), "P100PHX": set(),
                      "P101PHX": set(), "P102PHX": set(), "IDX#PHX": set()}
        self._quant = {"C98PHX": 0, "C99PHX": 0, "C100PHX": 0, "C101PHX": 0,
                       "C102PHX": 0, "P98PHX": 0, "P99PHX": 0, "P100PHX": 0,
                       "P101PHX": 0, "P102PHX": 0, "IDX#PHX": 0}
        self._filled = {}

    def _make_order(self, asset_code, quantity, base_price, spread, bid=True):
        return Order(asset_code = asset_code, quantity=quantity if bid else -1*quantity,
                     order_type = Order.ORDER_LMT,
                     price = base_price-spread/2 if bid else base_price+spread/2,
                     competitor_identifier = self._comp_id)

    def handle_exchange_update(self, exchange_update_response):
        print(exchange_update_response.competitor_metadata)
        print(self._quant)
        for order in self.latest_fills:
            if order.order.order_id not in self._filled:
                self._filled[order.order.order_id] = 0
            if order.order.quantity < 0:
                self._quant[order.order.asset_code] -= (order.filled_quantity-
                                                        self._filled[order.order.order_id])
                self._filled[order.order.order_id] = order.filled_quantity
            else:
                self._quant[order.order.asset_code] += (order.filled_quantity-
                                                        self._filled[order.order.order_id])
                self._filled[order.order.order_id] = order.filled_quantity
            self.cancel_order(order.order.order_id)
        fills = set([order.order.order_id for order in self.latest_fills])

        # remove fills
        for asset_code in self._bids:
            self._bids[asset_code] -= fills
        for asset_code in self._asks:
            self._asks[asset_code] -= fills

        for asset in exchange_update_response.market_updates:
            asset_code = asset.asset.asset_code

            # cancel previous MM orders if no takers
            if len(self._bids[asset_code]) >= 1 and len(self._asks[asset_code]) >= 1:
                while len(self._bids[asset_code]):
                    self.cancel_order(self._bids[asset_code].pop())
                while len(self._asks[asset_code]):
                    self.cancel_order(self._asks[asset_code].pop())

            bid_price = asset.bids[int(BID_PERCENTILE*len(asset.bids))].price
            ask_price = asset.asks[int(ASK_PERCENTILE*len(asset.asks))].price
            base_price = round((bid_price+ask_price)/2, 1)
            spread = round(ask_price-bid_price, 1)

            if self._quant[asset_code] < 10:
                bid_resp = self.place_order(self._make_order(asset_code, QUANTITY, base_price, spread, True))
                if type(bid_resp) != PlaceOrderResponse:
                    pass # print(bid_resp)
                else:
                    self._bids[asset_code].add(bid_resp.order_id)

            if self._quant[asset_code] > -10:
                ask_resp = self.place_order(self._make_order(asset_code, QUANTITY, base_price, spread, False))
                if type(ask_resp) != PlaceOrderResponse:
                    pass # print(ask_resp)
                else:
                    self._asks[asset_code].add(ask_resp.order_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the exchange client')
    parser.add_argument("--server_host", type=str, default="localhost")
    parser.add_argument("--server_port", type=str, default="50052")
    parser.add_argument("--client_id", type=str)
    parser.add_argument("--client_private_key", type=str)
    parser.add_argument("--websocket_port", type=int, default=5678)

    args = parser.parse_args()
    host, port, client_id, client_pk, websocket_port = (args.server_host, args.server_port,
                                        args.client_id, args.client_private_key,
                                        args.websocket_port)

    client = ExampleMarketMaker(host, port, client_id, client_pk, websocket_port)
    client.start_updates()
