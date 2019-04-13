import grpc
import asyncio
import argparse
import time
import threading
import json

from google.protobuf import text_format, json_format
from websocket_server import WebsocketServer

from protos.order_book_pb2 import (
    Order
)

from protos.service_pb2 import (
    RegisterCompetitorRequest,
    RegisterCompetitorResponse,
    CompetitorIdentifier,
    GetCompetitorMetadataRequest,
    GetExchangeUpdateRequest,
    PlaceOrderRequest,
    CancelOrderRequest,
    ModifyOrderRequest
)

from protos.service_pb2_grpc import ExchangeServiceStub

# frequency at which to update the websocket for the web dashboard.
# note that this *will* block receiving updates, so the faster we update
# the websocket the slower your code will run. additionally, if the frontend
# dashboard receives websocket messages too fast it may be slow.
_WEBSOCKET_UPDATE_FREQUENCY = 0.5

class BaseExchangeServerClient(threading.Thread):
    def __init__(self, server_host, server_port, competitor_id, competitor_pk,
                       websocket_port, start_websocket=True):
        if start_websocket:
            # start the thread
            self.websocket_port = websocket_port
            threading.Thread.__init__(self)
            self.start()

        # init grpc channel
        channel = grpc.insecure_channel('%s:%s' % (server_host, server_port))
        self._stub = ExchangeServiceStub(channel)

        # get the initial competitor id
        self._comp_id = CompetitorIdentifier(
            competitor_id = competitor_id,
            competitor_private_key = competitor_pk
        )

        # register the competitor
        self._register_competitor()

        self.latest_exchange_updates = None
        self.latest_fills = []
        self.latest_competitor_metadata = None

    def run(self):
        """Starts the websocket server"""
        # important to start the websocket server in run as opposed to init,
        # otherwise we'll start it in the main thread and block other requests
        # from coming in
        self._server = WebsocketServer(self.websocket_port, host='0.0.0.0')
        self._server.set_fn_new_client(self.new_client)
        self._server.run_forever()

    def new_client(self, client, server):
        while True:
            if self.latest_exchange_updates is not None:
                msg = {
                    'marketUpdates': [json.loads(json_format.MessageToJson(update))
                                        for update in self.latest_exchange_updates],
                    'clientId': self._comp_id.competitor_id,
                    'fills': [json.loads(json_format.MessageToJson(fill))
                                for fill in self.latest_fills]
                }

                if self.latest_competitor_metadata:
                    msg['competitorMetadata'] = json.loads(json_format.MessageToJson(self.latest_competitor_metadata))

                server.send_message(client, json.dumps(msg))

                # clear the set of latest fills
                self.latest_fills = []

                time.sleep(_WEBSOCKET_UPDATE_FREQUENCY)

    def start_updates(self):
        """Starts receiving updates from the exchange server"""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._get_exchange_updates())

    def handle_exchange_update(self, exchange_update_response):
        """Handler for exchange updates
        @param update (GetExchangeUpdateResponse)
        """
        raise NotImplementedError("Implement handle_exchange_update")

    def get_competitor_metadata(self):
        """Gets competitor metadata for this both
        @return GetCompetitorMetadataResponse
        """
        return self._stub.GetCompetitorMetadata(GetCompetitorMetadataRequest(
            competitor_identifier = self._comp_id
        ))

    def place_order(self, order: Order):
        try:
            r = self._stub.PlaceOrder(PlaceOrderRequest(competitor_identifier=self._comp_id,
                order=order))
            return r
        except Exception as e:
            return e

    def modify_order(self, order_id: str, new_order: Order):
        try:
            r = self._stub.ModifyOrder(ModifyOrderRequest(competitor_identifier=self._comp_id,
                order_id=order_id, new_order=new_order))
            return r
        except Exception as e:
            return e

    def cancel_order(self, order_id):
        try:
            r = self._stub.CancelOrder(CancelOrderRequest(competitor_identifier=self._comp_id,
                order_id=order_id))
            return r
        except Exception as e:
            return e

    def _register_competitor(self):
        self._stub.RegisterCompetitor(RegisterCompetitorRequest(
            competitor_identifier = self._comp_id
        ))

    async def _get_exchange_updates(self):
        req = GetExchangeUpdateRequest(competitor_identifier=self._comp_id)
        for exchange_update_response in self._stub.GetExchangeUpdate(req):
            # update exchange updates for sending to websocket
            self.latest_exchange_updates = exchange_update_response.market_updates

            if exchange_update_response.fills:
                self.latest_fills += exchange_update_response.fills

            if exchange_update_response.competitor_metadata:
                self.latest_competitor_metadata = exchange_update_response.competitor_metadata

            self.handle_exchange_update(exchange_update_response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the exchange client')
    parser.add_argument("--server_host", type=str, default="localhost")
    parser.add_argument("--server_port", type=str, default="50052")
    parser.add_argument("--client_id", type=str)
    parser.add_argument("--client_private_key", type=str)
    parser.add_argument("--websocket_port", type=int, default=5432)

    args = parser.parse_args()
    host, port, client_id, client_pk, websocket_port = (args.server_host, args.server_port,
                                        args.client_id, args.client_private_key,
                                        args.websocket_port)

    client = BaseExchangeServerClient(host, port, client_id, client_pk, websocket_port)
    client.start()
