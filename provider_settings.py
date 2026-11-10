"""Local model profiles. Credentials never appear in public settings."""
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
import engine

FILE = engine.ROOT / '.model-settings.json'
GO_URL = 'https://opencode.ai/zen/go/v1'
GO_MODEL = 'deepseek-v4.1-flash'


def opencode_local():
    try:
        data = json.loads((Path.home()/'.config/opencode/opencode.json').read_text())
        options = data['provider']['opencode']['options']
        if options.get('baseURL', '').rstrip('/') != GO_URL: return ''
        key = options.get('apiKey', '')
        return key if isinstance(key, str) and key and '{' not in key and '$' not in key else ''
    except (OSError, ValueError, KeyError, TypeError): return ''


def defaults():
    profiles={name:{'model':values[1], 'url':values[0]} for name,values in engine.PRESETS.items()}
    profiles['OpenCode Go']={'model':GO_MODEL,'url':GO_URL}
    return {'active':'硅基流动', 'profiles':profiles}


def read():
    data=defaults()
    if FILE.exists():
        try:
            saved=json.loads(FILE.read_text())
            if isinstance(saved,dict) and isinstance(saved.get('profiles'),dict):
                data['active']=saved.get('active', data['active'])
                data['profiles'].update(saved['profiles'])
        except (OSError,ValueError): raise ValueError('本地模型配置无法读取，请检查配置文件。')
    return data


def fallback(name):
    if name == 'OpenCode Go': return os.getenv('OPENCODE_GO_API_KEY','') or opencode_local()
    preset=engine.PRESETS.get(name)
    return (os.getenv(preset[2], '') or engine.local_api_key(preset[2])) if preset else ''


def public():
    data=read()
    return {'active':data['active'],'profiles':{name:{'model':p.get('model',''),'url':p.get('url',''),
            'key_configured':bool(p.get('key') or (fallback(name) if p.get('url','').rstrip('/') == default_url(name) else ''))}
            for name,p in data['profiles'].items()}}


def default_url(name):
    return GO_URL if name=='OpenCode Go' else engine.PRESETS.get(name, ('',))[0].rstrip('/')


def save(active, name, model, url, key):
    if name not in defaults()['profiles']: raise ValueError('不支持的服务商')
    if active not in defaults()['profiles']: raise ValueError('不支持的默认服务商')
    model=model.strip();url=url.strip().rstrip('/');key=key.strip()
    if not model or (not url and name!='Gemini'): raise ValueError('请填写模型名称和 API 地址')
    parts=urlsplit(url)
    if url and (parts.scheme not in ('http','https') or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment or (parts.scheme=='http' and parts.hostname not in ('localhost','127.0.0.1','::1'))):
        raise ValueError('远程 API 地址必须是无凭据和查询参数的 HTTPS 地址')
    data=read(); previous=data['profiles'].get(name,{})
    local = parts.scheme=='http' and parts.hostname in ('localhost','127.0.0.1','::1')
    if previous.get('url','').rstrip('/') != url and not key and not local:
        raise ValueError('更换 API 地址需要重新输入该地址专用密钥。')
    profile={'model':model,'url':url}
    if key: profile['key']=key
    elif previous.get('key'): profile['key']=previous['key']
    data['profiles'][name]=profile;data['active']=active
    FILE.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.model-settings-',dir=FILE.parent)
    try:
        os.fchmod(fd,0o600)
        with os.fdopen(fd,'w') as f: json.dump(data,f,ensure_ascii=False,indent=2)
        os.replace(tmp,FILE)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return public()


def resolve(name):
    data=read();name=name or data['active'];p=data['profiles'].get(name)
    if name not in defaults()['profiles'] or not p: raise ValueError('请先配置服务商')
    expected=default_url(name)
    key=p.get('key') or (fallback(name) if p.get('url','').rstrip('/') == expected else '')
    provider='自定义（OpenAI 兼容）' if name=='OpenCode Go' else name
    config=engine.connection(provider,p.get('model',''),p.get('url',''),key,allow_env_key=False)
    config['provider']=name
    return config
