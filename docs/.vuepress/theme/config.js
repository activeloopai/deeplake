const hasValue = target => target !== undefined && target !== null

const getWithPath = (object, path) => {
  const p = path.split('.')
  const length = p.length
  let index = 0

  while (hasValue(object) && index < length) {
    object = object[p[index++]]
  }

  return index && index === length ? object : undefined
}

const get = ($site, name, locale) => {
  if (locale) {
    return (
      getWithPath($site, `themeConfig.locales.${locale}.${name}`) ||
      getWithPath($site, `themeConfig.${name}`) ||
      getWithPath($site, `locales.${locale}.${name}`) ||
      $site[name]
    )
  }

  return getWithPath($site, `themeConfig.${name}`) || $site[name]
}

export default {
  get,
}
