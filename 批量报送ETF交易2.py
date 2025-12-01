import requests

cookies = {
    'sensorsdata2015jssdkcross': '%7B%22distinct_id%22%3A%22huangyufei%22%2C%22first_id%22%3A%22191c5455ab8186-0a0ee3b22e7e8f8-26001151-2073600-191c5455ab91da%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%2C%22%24latest_referrer%22%3A%22%22%7D%2C%22identities%22%3A%22eyIkaWRlbnRpdHlfY29va2llX2lkIjoiMTkxYzU0NTVhYjgxODYtMGEwZWUzYjIyZTdlOGY4LTI2MDAxMTUxLTIwNzM2MDAtMTkxYzU0NTVhYjkxZGEiLCIkaWRlbnRpdHlfbG9naW5faWQiOiJodWFuZ3l1ZmVpIn0%3D%22%2C%22history_login_id%22%3A%7B%22name%22%3A%22%24identity_login_id%22%2C%22value%22%3A%22huangyufei%22%7D%2C%22%24device_id%22%3A%22191c5455ab8186-0a0ee3b22e7e8f8-26001151-2073600-191c5455ab91da%22%7D',
    'LtpaToken2': '3+8fn8H071aoH6xsfj9qen5yA1CG3kOZnhjNmcJ6rYOB/BD9hSmADQFifcjbihUfbPxuBLGOmGASNJTwsuxEQl7OHdQkO4GIy4Hey4s+Z4Io2NmIS26+XS0XNmzLhM2J+x/Dg7hgPLfsCA+xn0N1qD9OF8hslACpw7d9HBBsMyFZQPNJR3MDoFYJL154J6eUMPgs1npiqaj1AEs0QARhK18T7t8XDk6bs0/TWXDgwwBLJCJJELBB7wMBr8O5D+9FaLMwoSxz3fbERs7CBcq7Ar3SySkQ+4PZnWHlU5WGRo8CfQIHFWbtBQBur5xwrypSVGT62cSIuibYTE/BMiiWhc6JkmeTBY6L/Lia5eScUlI=',
    'Admin-Token': 'eyJhbGciOiJIUzUxMiJ9.eyJsb2dpbl91c2VyX2tleSI6IjQzNGI0ODk4LTYzYWQtNDE1MC04YmVkLTUwZDA5NGFiMDA3MiJ9.ScDaFx6iz6EEK1DEvrFc6Bs6yBTjtJaTGWnv5JC0_HyuHtVOyVaMEhVx4XXM7UzQ1rx3ey-kPIncTp85r7AL4g',
    'oauth_token': '8ac57e84-02c2-4484-bdad-d219f1eec976',
    'login_type': 'oa',
}

headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Authorization': 'Bearer eyJhbGciOiJIUzUxMiJ9.eyJsb2dpbl91c2VyX2tleSI6IjQzNGI0ODk4LTYzYWQtNDE1MC04YmVkLTUwZDA5NGFiMDA3MiJ9.ScDaFx6iz6EEK1DEvrFc6Bs6yBTjtJaTGWnv5JC0_HyuHtVOyVaMEhVx4XXM7UzQ1rx3ey-kPIncTp85r7AL4g',
    'Connection': 'keep-alive',
    'Content-Type': 'application/json;charset=UTF-8',
    'Origin': 'http://alm2.gf.com.cn',
    'Referer': 'http://alm2.gf.com.cn/toolbox/toolbox/simulation',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
    # 'Cookie': 'sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%22huangyufei%22%2C%22first_id%22%3A%22191c5455ab8186-0a0ee3b22e7e8f8-26001151-2073600-191c5455ab91da%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%2C%22%24latest_referrer%22%3A%22%22%7D%2C%22identities%22%3A%22eyIkaWRlbnRpdHlfY29va2llX2lkIjoiMTkxYzU0NTVhYjgxODYtMGEwZWUzYjIyZTdlOGY4LTI2MDAxMTUxLTIwNzM2MDAtMTkxYzU0NTVhYjkxZGEiLCIkaWRlbnRpdHlfbG9naW5faWQiOiJodWFuZ3l1ZmVpIn0%3D%22%2C%22history_login_id%22%3A%7B%22name%22%3A%22%24identity_login_id%22%2C%22value%22%3A%22huangyufei%22%7D%2C%22%24device_id%22%3A%22191c5455ab8186-0a0ee3b22e7e8f8-26001151-2073600-191c5455ab91da%22%7D; LtpaToken2=3+8fn8H071aoH6xsfj9qen5yA1CG3kOZnhjNmcJ6rYOB/BD9hSmADQFifcjbihUfbPxuBLGOmGASNJTwsuxEQl7OHdQkO4GIy4Hey4s+Z4Io2NmIS26+XS0XNmzLhM2J+x/Dg7hgPLfsCA+xn0N1qD9OF8hslACpw7d9HBBsMyFZQPNJR3MDoFYJL154J6eUMPgs1npiqaj1AEs0QARhK18T7t8XDk6bs0/TWXDgwwBLJCJJELBB7wMBr8O5D+9FaLMwoSxz3fbERs7CBcq7Ar3SySkQ+4PZnWHlU5WGRo8CfQIHFWbtBQBur5xwrypSVGT62cSIuibYTE/BMiiWhc6JkmeTBY6L/Lia5eScUlI=; Admin-Token=eyJhbGciOiJIUzUxMiJ9.eyJsb2dpbl91c2VyX2tleSI6IjQzNGI0ODk4LTYzYWQtNDE1MC04YmVkLTUwZDA5NGFiMDA3MiJ9.ScDaFx6iz6EEK1DEvrFc6Bs6yBTjtJaTGWnv5JC0_HyuHtVOyVaMEhVx4XXM7UzQ1rx3ey-kPIncTp85r7AL4g; oauth_token=8ac57e84-02c2-4484-bdad-d219f1eec976; login_type=oa',
}

json_data = {
    'tradingAmount': 1,
    'targetCode': '513110.SH',
    'tradingPosition': '1',
    'tradingDirection': '1',
    'tradingDate': '2025-10-15',
    'tradingDescription': '1',
    'groupId': '1901475930600738818',
}

response = requests.post(
    'http://alm2.gf.com.cn/alm-api/simulateTrading/transaction/save',
    cookies=cookies,
    headers=headers,
    json=json_data,
    verify=False,
)

print(response.text)