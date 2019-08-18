<template>
  <div class="curl__container">
    <div ref="curl">
      <slot></slot>
    </div>
    <Button :light="lightBtn" :loading="loading" @click="sendRequest">Send request now</Button>
  </div>
</template>

<script>
import nprogress from 'nprogress'

import request from '../../request'
import { isJSON, isQueryString, parseQueryString } from '../../utils'
import curl from '../../curl'

// configure progress bar
nprogress.configure({ showSpinner: false })

export default {
  name: 'CURL',
  data() {
    return {
      lightBtn: true,
      loading: false,
    }
  },
  methods: {
    sendRequest() {
      const cmd = this.$refs.curl.outerText.trim()

      if (this.loading) return

      const options = curl(cmd)

      if (!options) {
        this.$message.error(
          'Got an invalid CURL command, please check it and try again.'
        )
        return
      }

      console.clear()
      console.log('====== DEBUG INFO ======')
      console.info(`=> ${options.method.toUpperCase()} ${options.url}`)

      if (options.headers) {
        console.info('=> Headers:', options.headers)
      }

      if (options.params) {
        console.info('=> Params:', options.params)
      }

      if (options.data) {
        console.info('=> Data:', options.data)
      }

      this.openLoading()

      request(options)
        .then(data => {
          this.closeLoading()
          this.$message.success(
            'Request success. Open console to get more details.'
          )

          console.info('<=', data)
        })
        .catch(err => {
          this.closeLoading()
          this.$message.error(
            `${err.status} ${err.message}. Open console to get more details.`
          )

          console.error('<=', err)
        })
    },
    notInExampleBox() {
      return this.$el.parentNode.getAttribute('type') !== 'example'
    },
    openLoading() {
      this.loading = true

      nprogress.start()
    },
    closeLoading() {
      this.loading = false

      nprogress.done()
    },
  },
  mounted() {
    if (this.notInExampleBox()) {
      this.lightBtn = false
    }
  },
}
</script>
