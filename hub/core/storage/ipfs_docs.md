## IPFS Storage Provider for Activeloop Hub

IPFS (Inter Planetary File System) is a collection of protocols that together form the storage layer of the [decentralized web](https://en.wikipedia.org/wiki/Decentralized_web). The decentralized web offers new ways to store and interact with data, which quite naturally lends itself to usage for decentralized AI development. Here, we present the integration of decentralized storage with Activeloop Hub for storing and accessing machine learning datasets on IPFS.

### IPFSProvider

The core addition to the Hub is the `IPFSProvider` class, which is integrated within the rest of the `core` library. To start using the `IPFSProvider` in your Activeloop stack, you need to input the following parameters:
- `coreurl`: the URL that specifies the IPFS Gateway, used for storing files on IPFS (for more details on gateways, see [this documentation](https://docs.ipfs.io/concepts/ipfs-gateway/#overview))
- `storage_type`: specifies which gateway or other provider you are using, needs to be one of `local`, `infura`, `estuary`, `web3.storage`, `pinata`
- `api_key`: [optional] if you provided a `coreurl` to a private provider, you need to add your api key to access their storage services (applies to `estuary`, `web3.storage`, `pinata`)