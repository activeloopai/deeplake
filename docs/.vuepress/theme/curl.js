import { isJSON, isQueryString, parseQueryString } from './utils'

const parseField = s => {
  return s.split(/: (.+)/)
}

export function isURL(url) {
  if (!url || url.length >= 2083 || /[\s<>]/.test(url)) return false

  // ensure URL starts with HTTP/HTTPS
  const urlRegexp = /^(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})/
  const localhost = /^(https?:\/\/)?localhost(\:\d+)*(?:[^\:?\d]\S*)?$/

  return urlRegexp.test(url) || localhost.test(url)
}

export function isCURL(cmd) {
  return cmd.startsWith('curl ')
}

export default function curl(cmd) {
  if (!isCURL(cmd)) return

  const args = cmd
    .match(/"[^"]+"|'[^']+'|\S+/g)
    .filter(item => item.trim() !== '\\')

  const result = {
    method: 'GET',
    headers: {},
  }

  let state = ''

  args.forEach(function(arg) {
    switch (true) {
      case isURL(arg):
        result.url = arg

        break
      case arg === '-A' || arg === '--user-agent':
        state = 'user-agent'

        break
      case arg === '-H' || arg === '--header':
        state = 'header'

        break
      case arg === '-d' || arg === '--data' || arg === '--data-ascii':
        state = 'data'

        break
      case arg === '-u' || arg === '--user':
        state = 'user'

        break
      case arg === '-I' || arg === '--head':
        result.method = 'HEAD'

        break
      case arg === '-X' || arg === '--request':
        state = 'method'

        break
      case arg === '-b' || arg === '--cookie':
        state = 'cookie'

        break
      case arg === '--compressed':
        result.headers['Accept-Encoding'] =
          result.headers['Accept-Encoding'] || 'deflate, gzip'

        break
      case !!arg:
        // Delete the start position and the end of the quotation mark
        if (/^['"]/.test(arg)) {
          arg = arg.slice(1, -1)
        }

        switch (state) {
          case 'header':
            const field = parseField(arg)

            result.headers[field[0]] = field[1]
            state = ''

            break
          case 'user-agent':
            result.headers['User-Agent'] = arg
            state = ''

            break
          case 'data':
            if (
              result.method.toUpperCase() === 'GET' ||
              result.method.toUpperCase() === 'HEAD'
            ) {
              result.method = 'POST'
            }

            result.headers['Content-Type'] =
              result.headers['Content-Type'] || 'application/json'

            if (isJSON(arg)) {
              const data = JSON.parse(arg)

              result.data = result.data
                ? Object.assign(result.data, data)
                : data
            } else if (isQueryString(arg)) {
              result.data = result.data ? result.data + '&' + arg : arg
            }

            state = ''

            break
          case 'user':
            result.headers['Authorization'] = 'Basic ' + btoa(arg)
            state = ''

            break
          case 'method':
            result.method = arg.toLowerCase()
            state = ''

            break
          case 'cookie':
            result.headers['Set-Cookie'] = arg
            state = ''

            break
        }
        break
    }
  })

  return result
}
